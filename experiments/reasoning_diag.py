"""Diagnostic: is k-hop pointer-chasing learnable AT ALL on this 0.8B?

Easier setting (5 symbols, k=1 only, more steps) + PRINT actual model outputs, to tell
apart: (a) task too hard, (b) training collapse (constant output), (c) eval/tokenization
bug. Once base+LoRA can do the simplest case, we scale back up.

Run:  PYTHONPATH=../src python reasoning_diag.py
"""
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt

rt.SYMS = list("abcde")          # 5 symbols -> chance = 20%


def gen1(m, tok, prompt):
    text = tok.apply_chat_template([{"role": "system", "content": rt.RSYS}, {"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids]).to(g.DEV)
    out = ids
    for _ in range(4):
        with torch.no_grad(), torch.autocast(g.DEV, dtype=torch.bfloat16):
            nxt = m(input_ids=out, use_cache=False).logits[:, -1, :].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    data = rt.make_data(2000, ks=[1], seed=0)          # k=1 only, simplest
    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
    m = g.add_lora(m, 16)
    rt.train(m, data, tok, steps=3000, lr=2e-4)
    print("acc (5 syms):", rt.acc_by_k(m, tok, [1, 2, 3], n=60), "  (chance=20%)", flush=True)
    print("--- sample k=1 outputs (single lookup) ---", flush=True)
    rng = random.Random(999)
    for _ in range(8):
        p, a = rt.make_example(rng, 1)
        print(f"  {p[5:].rsplit('.', 2)[0]:<60} want {a!r}  got {gen1(m, tok, p)!r}", flush=True)


if __name__ == "__main__":
    main()
