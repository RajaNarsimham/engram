"""Does ADDED DEPTH (output-head) let the 0.8B do a hop it couldn't?

base+LoRA fails fixed 2-hop (~25% = chance). Test whether adding a trainable
attention output-head (extra depth: base does hop 1, head does hop 2) crosses it
from chance -> learned. If head x2/x4 succeed where plain LoRA failed, the added
depth bought the reasoning hop -- the thesis, on the task that can show it.

Run:  PYTHONPATH=../src python reasoning_head.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt                 # rt.SYMS = 'abcde'
from grow_hybrid import add_lora_and_head


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    K = 2
    data = rt.make_data(2000, ks=[K], seed=0)        # FIXED 2-hop
    configs = [
        ("base+LoRA (no head)", lambda m: g.add_lora(m, 16)),
        ("base+LoRA+head x2",   lambda m: add_lora_and_head(m, 16, 2)),
        ("base+LoRA+head x4",   lambda m: add_lora_and_head(m, 16, 4)),
    ]
    for label, build in configs:
        print(f"--- {label} ---", flush=True)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = build(m)
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rt.train(m, data, tok, epochs=12)        # batched -> GPU actually fills
        acc = rt.acc_by_k(m, tok, [K], n=80)
        print(f"   {label}: fixed k={K} acc {acc[K]}%  (chance 20%, base+LoRA was ~25%) | trainable {ntr/1e6:.1f}M", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
