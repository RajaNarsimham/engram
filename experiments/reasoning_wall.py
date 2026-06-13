"""Find the REAL reasoning-depth wall of base+LoRA on the 0.8B, with adequate
(batched) training. We now know k=1 and k=2 = 100%; sweep fixed k=3..7 to find
where it actually breaks. THAT wall is the baseline the deeper arms must beat.

Run:  PYTHONPATH=../src python reasoning_wall.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt           # rt.SYMS = 'abcde'


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    rows = []
    for k in (3, 4, 5, 6, 7):
        data = rt.make_data(2000, ks=[k], seed=0)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = g.add_lora(m, 16)
        rt.train(m, data, tok, epochs=10)
        acc = rt.acc_by_k(m, tok, [k], n=80)[k]
        rows.append((k, acc))
        print(f"FIXED k={k}: base+LoRA acc {acc}%", flush=True)
        del m
        torch.cuda.empty_cache()
    print("\n=== base+LoRA reasoning wall (k1=100, k2=100 already) ===", flush=True)
    for k, acc in rows:
        print(f"  k={k}: {acc}%", flush=True)


if __name__ == "__main__":
    main()
