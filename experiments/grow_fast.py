"""Fast eval + confirmation of the hybrid result.

fast_recall replaces the slow token-by-token generation (120 facts x 16 steps, no
cache) with a few BATCHED teacher-forced forward passes: build [prompt+answer],
right-pad (causal attention => no pad leakage at answer positions, so the output-head
is fine), one forward per chunk, and check whether the model greedily predicts the
answer tokens. ~4 forwards total instead of ~1920.

Run:  PYTHONPATH=../src python grow_fast.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
from grow_head import add_head
from grow_hybrid import add_lora_and_head

DEV = g.DEV


def fast_recall(model, tok, bs=30):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = []
    for q, a, _, _ in g.FACTS:
        ptext = tok.apply_chat_template([{"role": "system", "content": g.SYS}, {"role": "user", "content": q}],
                                        add_generation_prompt=True, tokenize=False)
        pid = tok(ptext, add_special_tokens=False).input_ids
        aid = tok(a, add_special_tokens=False).input_ids
        items.append((pid + aid, len(pid), len(pid) + len(aid)))
    ok = 0
    for c in range(0, len(items), bs):
        chunk = items[c:c + bs]
        maxlen = max(len(s) for s, _, _ in chunk)
        ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, (s, _, _) in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1                       # RIGHT pad -> causal keeps answer positions clean
        ids, attn = ids.to(DEV), attn.to(DEV)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
        for i, (s, a0, a1) in enumerate(chunk):
            if torch.equal(logits[i, a0 - 1:a1 - 1].argmax(-1), ids[i, a0:a1]):
                ok += 1
    return round(100 * ok / len(items))


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    configs = [
        ("LoRA r16 only",       lambda m: g.add_lora(m, 16)),
        ("head x2 only",        lambda m: add_head(m, 2)),
        ("LoRA r16 + head x2",  lambda m: add_lora_and_head(m, 16, 2)),
    ]
    rows = []
    for label, build in configs:
        print(f"--- {label} ---", flush=True)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
        m = build(m)
        g.train(m, 1500, 2e-4, tok)
        rec = fast_recall(m, tok)
        rows.append((label, rec))
        print(f"   {label}: recall {rec}% (fast/teacher-forced)", flush=True)
        del m
        torch.cuda.empty_cache()
    print("\n=== HYBRID confirmation (fast eval) ===", flush=True)
    for label, rec in rows:
        print(f"{label:<22}{rec:>4}%", flush=True)


if __name__ == "__main__":
    main()
