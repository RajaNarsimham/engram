"""Warm-started added depth, the RIGHT way: append copies of the base's OWN layers
at the top (real weights, output scaled x0.1 -> near-identity but competent + gradients
flow), vs the dead zero-init we used before. Test at k=8 (12 syms) where base+LoRA ~36-40%.

If +2/+4 warm-started base-layer copies cross the wall, depth-done-right buys reasoning.

Run:  PYTHONPATH=../src python reasoning_warmtop.py
"""
import copy

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt

rt.SYMS = list("abcdefghijkl")          # 12 symbols


def extend_top(m, n, scale=0.1):
    """Append copies of the last n base layers; scale their output projections by
    `scale` so they start near-identity but with REAL (competent) weights."""
    layers = list(m.model.language_model.layers)
    cfg = m.model.language_model.config
    new = []
    for L in layers[-n:]:
        c = copy.deepcopy(L)
        with torch.no_grad():
            if hasattr(c, "linear_attn"):
                c.linear_attn.out_proj.weight.mul_(scale)
            if hasattr(c, "self_attn"):
                c.self_attn.o_proj.weight.mul_(scale)
            c.mlp.down_proj.weight.mul_(scale)
        new.append(c)
    allL = layers + new
    m.model.language_model.layers = torch.nn.ModuleList(allL)
    cfg.num_hidden_layers = len(allL)
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = list(cfg.layer_types) + list(cfg.layer_types)[-n:]
    for i, L in enumerate(allL):
        for an in ("self_attn", "linear_attn"):
            if hasattr(L, an):
                setattr(getattr(L, an), "layer_idx", i)
    return m


def add_warm_top(m, n):
    m = extend_top(m, n)
    return g.add_lora(m, 16)


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    K = 8
    data = rt.make_data(1500, ks=[K], seed=0)
    configs = [
        ("base+LoRA (24L)",        lambda m: g.add_lora(m, 16)),
        ("+2 warm layers (top)",   lambda m: add_warm_top(m, 2)),
        ("+4 warm layers (top)",   lambda m: add_warm_top(m, 4)),
    ]
    rows = []
    for label, build in configs:
        print(f"--- {label} ---", flush=True)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = build(m)
        rt.train(m, data, tok, epochs=12, bs=16)
        acc = rt.acc_by_k(m, tok, [K], n=80)[K]
        rows.append((label, acc))
        print(f"   {label}: k={K} acc {acc}%", flush=True)
        del m
        torch.cuda.empty_cache()
    print(f"\n=== WARM-STARTED added depth at k={K} (12 syms, base+LoRA ~36-40%) ===", flush=True)
    for label, acc in rows:
        print(f"  {label:<24}{acc}%", flush=True)


if __name__ == "__main__":
    main()
