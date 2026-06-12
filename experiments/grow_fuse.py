"""Parallel logit FUSION instead of series (the user's fix for the head bottleneck).

Train LoRA and an output-head INDEPENDENTLY on the frozen base, then fuse their logits:
  fused = logits_lora + w * logits_head
All towers share the same base vocab, so logits align exactly. Sweep w. If any w beats
LoRA-alone, the head is COMPLEMENTARY (adds facts LoRA missed); if not, redundant.
Compare to the SERIES result (LoRA->head = 48%, which HURT LoRA's 63%).

Run:  PYTHONPATH=../src python grow_fuse.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
from grow_head import add_head

DEV = g.DEV


def build_items(tok):
    items = []
    for q, a, _, _ in g.FACTS:
        ptext = tok.apply_chat_template([{"role": "system", "content": g.SYS}, {"role": "user", "content": q}],
                                        add_generation_prompt=True, tokenize=False)
        pid = tok(ptext, add_special_tokens=False).input_ids
        aid = tok(a, add_special_tokens=False).input_ids
        items.append((pid + aid, len(pid), len(pid) + len(aid)))
    return items


def answer_logits(model, tok, items, bs=30):
    """Per fact: (logits at the answer positions [La,V] on CPU, true answer ids [La])."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    res = []
    for c in range(0, len(items), bs):
        chunk = items[c:c + bs]
        maxlen = max(len(s) for s, _, _ in chunk)
        ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, (s, _, _) in enumerate(chunk):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            lg = model(input_ids=ids.to(DEV), attention_mask=attn.to(DEV), use_cache=False).logits
        for i, (s, a0, a1) in enumerate(chunk):
            res.append((lg[i, a0 - 1:a1 - 1].float().cpu(), ids[i, a0:a1].clone()))
    return res


def score(lst):
    return round(100 * sum(torch.equal(lg.argmax(-1), t) for lg, t in lst) / len(lst))


def score_fused(la, lh, w):
    return round(100 * sum(torch.equal((a + w * h).argmax(-1), t) for (a, t), (h, _) in zip(la, lh)) / len(la))


def trained_logits(tok, items, build, name):
    print(f"training {name}...", flush=True)
    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
    m = build(m)
    g.train(m, 1500, 2e-4, tok)
    lg = answer_logits(m, tok, items)
    del m
    torch.cuda.empty_cache()
    return lg


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    items = build_items(tok)
    la = trained_logits(tok, items, lambda m: g.add_lora(m, 16), "LoRA")
    lh = trained_logits(tok, items, lambda m: add_head(m, 2), "head")

    print(f"\nLoRA alone:  {score(la)}%", flush=True)
    print(f"head alone:  {score(lh)}%", flush=True)
    print("(series LoRA->head was 48%, which HURT LoRA)", flush=True)
    print("--- PARALLEL FUSION  fused = logits_lora + w * logits_head ---", flush=True)
    best = (0, 0)
    for w in (0.25, 0.5, 1.0, 2.0, 4.0):
        s = score_fused(la, lh, w)
        best = max(best, (s, w))
        print(f"  w={w:<4}:  {s}%", flush=True)
    print(f"\nbest fusion: {best[0]}% at w={best[1]}  (LoRA alone {score(la)}%)", flush=True)


if __name__ == "__main__":
    main()
