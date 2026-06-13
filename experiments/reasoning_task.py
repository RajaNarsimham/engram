"""Depth-bottlenecked reasoning task: k-hop pointer-chasing, answered DIRECTLY (no CoT).

Given a fresh mapping (each letter -> one letter) IN THE PROMPT and a start letter,
follow the arrows k times and output the final letter. The map is new every example,
so it can't be memorized -- the model must COMPUTE the k-step traversal. Answering
directly (no chain-of-thought) makes it DEPTH-bounded: a transformer can do ~depth
sequential steps per forward pass, so accuracy should hold at low k and fall as k grows.

This run establishes the probe: base (untrained, ~chance) and base+LoRA (shallow
adaptation). If LoRA is high at low k and DROPS at high k, depth is the bottleneck and
the task is valid -- then we add the deeper arms (head / looped / doubled).

Run:  PYTHONPATH=../src python reasoning_task.py
"""
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g

DEV = g.DEV
SYMS = list("abcde")                            # 5 symbols -> chance = 20% (10 was too hard for 0.8B)
RSYS = "Follow the mapping the given number of steps. Output only the final letter, nothing else."


def make_example(rng, k):
    perm = SYMS[:]
    rng.shuffle(perm)
    mp = {s: perm[i] for i, s in enumerate(SYMS)}
    start = rng.choice(SYMS)
    cur = start
    for _ in range(k):
        cur = mp[cur]
    mstr = ", ".join(f"{s}->{mp[s]}" for s in SYMS)
    return f"Map: {mstr}. Start at {start}, take {k} steps.", cur


def make_data(n, ks, seed):
    rng = random.Random(seed)
    return [make_example(rng, rng.choice(ks)) for _ in range(n)]


def encode(tok, prompt, answer):
    text = tok.apply_chat_template([{"role": "system", "content": RSYS}, {"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    pid = tok(text, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    return torch.tensor([pid + aid]), torch.tensor([[-100] * len(pid) + aid])


def train(model, data, tok, epochs=15, lr=2e-4, bs=32):
    """Batched training (right-pad + attention_mask + label masking). ~10-20x faster
    than batch-size-1 because the GPU actually fills up."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr)
    enc = []
    for prompt, answer in data:
        text = tok.apply_chat_template([{"role": "system", "content": RSYS}, {"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        pid = tok(text, add_special_tokens=False).input_ids
        aid = tok(answer, add_special_tokens=False).input_ids
        enc.append((pid + aid, [-100] * len(pid) + aid))
    rng = random.Random(1)
    model.train()
    for _ in range(epochs):
        rng.shuffle(enc)
        for c in range(0, len(enc), bs):
            batch = enc[c:c + bs]
            B, ml = len(batch), max(len(ids) for ids, _ in batch)
            input_ids = torch.full((B, ml), pad, dtype=torch.long)
            labels = torch.full((B, ml), -100, dtype=torch.long)
            attn = torch.zeros((B, ml), dtype=torch.long)
            for i, (ids, lab) in enumerate(batch):
                input_ids[i, :len(ids)] = torch.tensor(ids)
                labels[i, :len(lab)] = torch.tensor(lab)
                attn[i, :len(ids)] = 1
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = model(input_ids=input_ids.to(DEV), attention_mask=attn.to(DEV),
                             labels=labels.to(DEV), use_cache=False).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()


def acc_by_k(model, tok, ks, n=80, bs=40, seed=123):
    rng = random.Random(seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = {}
    for k in ks:
        items = []
        for _ in range(n):
            p, a = make_example(rng, k)
            text = tok.apply_chat_template([{"role": "system", "content": RSYS}, {"role": "user", "content": p}],
                                           add_generation_prompt=True, tokenize=False)
            pid = tok(text, add_special_tokens=False).input_ids
            aid = tok(a, add_special_tokens=False).input_ids
            items.append((pid + aid, len(pid), len(pid) + len(aid)))
        ok = 0
        for c in range(0, len(items), bs):
            chunk = items[c:c + bs]
            ml = max(len(s) for s, _, _ in chunk)
            ids = torch.full((len(chunk), ml), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), ml), dtype=torch.long)
            for i, (s, _, _) in enumerate(chunk):
                ids[i, :len(s)] = torch.tensor(s)
                attn[i, :len(s)] = 1
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
                lg = model(input_ids=ids.to(DEV), attention_mask=attn.to(DEV), use_cache=False).logits
            for i, (s, a0, a1) in enumerate(chunk):
                if torch.equal(lg[i, a0 - 1:a1 - 1].argmax(-1).cpu(), ids[i, a0:a1]):
                    ok += 1
        out[k] = round(100 * ok / len(items))
    return out


def fmt(d):
    return "  ".join(f"k{k}:{v}%" for k, v in d.items())


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    KS = [1, 2, 3, 4, 5, 6]
    data = make_data(3000, ks=[1, 2, 3, 4, 5], seed=0)     # train on a RANGE so it respects step count

    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
    print("base (untrained):  " + fmt(acc_by_k(m, tok, KS)) + "   (chance 20%)", flush=True)
    del m
    torch.cuda.empty_cache()

    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
    m = g.add_lora(m, 16)
    train(m, data, tok, steps=5000)
    print("base + LoRA r16:   " + fmt(acc_by_k(m, tok, KS)), flush=True)


if __name__ == "__main__":
    main()
