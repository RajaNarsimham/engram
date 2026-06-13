"""5 symbols was too easy (no wall to k=7). Push to 12 symbols (longer chains, no
trivial short cycles) and test k=4/8/12. If accuracy DECLINES with k -> a real depth
wall to test against. If it stays flat-high -> pointer-chasing is parallelizable and
we switch to a provably-sequential task (state-tracking).

Run:  PYTHONPATH=../src python reasoning_wall2.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt

rt.SYMS = list("abcdefghijkl")          # 12 symbols -> chance ~8%


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    rows = []
    for k in (4, 8, 12):
        data = rt.make_data(2000, ks=[k], seed=0)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = g.add_lora(m, 16)
        rt.train(m, data, tok, epochs=10, bs=24)
        acc = rt.acc_by_k(m, tok, [k], n=80)[k]
        rows.append((k, acc))
        print(f"FIXED k={k} (12 syms): base+LoRA acc {acc}%", flush=True)
        del m
        torch.cuda.empty_cache()
    print("\n=== base+LoRA, 12 symbols (chance ~8%) ===", flush=True)
    for k, acc in rows:
        print(f"  k={k}: {acc}%", flush=True)


if __name__ == "__main__":
    main()
