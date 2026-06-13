"""Fixed-k depth probe: train a model to ALWAYS do exactly k hops (no step-count
conditioning, which the 0.8B couldn't learn). The largest k it can learn ~ its
reasoning depth. We know k=1=100%; find where base+LoRA breaks. Then deeper arms
try to push the wall higher.

Run:  PYTHONPATH=../src python reasoning_fixedk.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt          # rt.SYMS already = 'abcde'


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    for k in (2, 3, 4):
        data = rt.make_data(2000, ks=[k], seed=0)        # FIXED k only
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = g.add_lora(m, 16)
        rt.train(m, data, tok, steps=3000)
        acc = rt.acc_by_k(m, tok, [k], n=80)
        print(f"FIXED k={k}: base+LoRA acc {acc[k]}%   (chance 20%)", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
