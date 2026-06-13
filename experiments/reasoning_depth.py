"""THE depth test. At the wall (k=8, 12 symbols) where base+LoRA = ~40%, do DEEPER
models do better, with equal training? If yes, depth buys reasoning.

  base+LoRA (24L)            -- baseline (the wall)
  base+LoRA+head x4 (24+4)   -- the user's top-depth mechanism
  doubled+LoRA (48L)         -- genuine 2x depth

All trained equally (controls for the undertraining confound).
Run:  PYTHONPATH=../src python reasoning_depth.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt
from grow_hybrid import add_lora_and_head
from grow_lib import double_depth

rt.SYMS = list("abcdefghijkl")          # 12 symbols


def add_double_lora(m, r):
    m, _ = double_depth(m)
    return g.add_lora(m, r)


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    K = 8
    data = rt.make_data(1500, ks=[K], seed=0)
    configs = [
        ("base+LoRA (24L)",          lambda m: g.add_lora(m, 16)),
        ("base+LoRA+head x4 (24+4)",  lambda m: add_lora_and_head(m, 16, 4)),
        ("doubled+LoRA (48L)",        lambda m: add_double_lora(m, 16)),
    ]
    rows = []
    for label, build in configs:
        print(f"--- {label} ---", flush=True)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = build(m)
        rt.train(m, data, tok, epochs=12, bs=16)
        acc = rt.acc_by_k(m, tok, [K], n=80)[K]
        rows.append((label, acc))
        print(f"   {label}: k={K} (12 syms) acc {acc}%", flush=True)
        del m
        torch.cuda.empty_cache()
    print(f"\n=== DEPTH TEST at the wall (k={K}, 12 syms, base+LoRA was ~40%) ===", flush=True)
    for label, acc in rows:
        print(f"  {label:<26}{acc}%", flush=True)


if __name__ == "__main__":
    main()
