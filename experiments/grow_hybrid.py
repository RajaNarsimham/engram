"""Hybrid: LoRA adapters BETWEEN layers + a trainable output HEAD on top.

LoRA adapts the base's own layers (storage/rank); the head adds stable top-depth and
now sees LoRA-ADAPTED features (not frozen). Tests whether the two are complementary
or redundant, on the same 120-fact benchmark.

Run:  PYTHONPATH=../src python grow_hybrid.py
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
from grow_head import add_head

DEV = g.DEV


def lora_targets(m):
    return sorted({n.split(".")[-1] for n, mm in m.named_modules()
                   if isinstance(mm, torch.nn.Linear) and ".language_model." in n and "lora" not in n.lower()})


def add_lora_and_head(m, r, n):
    m = add_head(m, n)                       # head + pre-hook on lm_head; head trainable
    head = m._head
    m = get_peft_model(m, LoraConfig(target_modules=lora_targets(m), r=r, lora_alpha=2 * r,
                                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    for p in head.parameters():              # PEFT froze everything; re-enable the head
        p.requires_grad = True
    return m


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
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        g.train(m, 1500, 2e-4, tok)
        rec = g.recall(m, tok)
        rows.append((label, rec, ntr))
        print(f"   {label}: recall {rec}% | trainable {ntr/1e6:.1f}M", flush=True)
        del m
        torch.cuda.empty_cache()

    print("\n=== HYBRID (LoRA + head) on 120 facts ===", flush=True)
    for label, rec, ntr in rows:
        print(f"{label:<22}{rec:>4}%   {ntr/1e6:>6.1f}M", flush=True)


if __name__ == "__main__":
    main()
