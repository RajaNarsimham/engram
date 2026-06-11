"""Config 3, done faithfully: insert genuine CHEAP TRANSFORMER LAYERS -- low-rank
ATTENTION (token mixing) + low-rank gated MLP -- after each existing layer, instead
of the MLP-only bottleneck adapters. Output projections zero-init => identity at start
(function-preserving), then trained. Same 120-fact benchmark as grow_experiment.

Run:  PYTHONPATH=../src python grow_attn.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g

DEV = g.DEV


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        n = x.float()
        n = n / (n.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        return (self.w * n).to(x.dtype)


class LowRankLayer(nn.Module):
    """A real transformer layer with everything low-rank (d->r->d), causal single-head
    attention + gated MLP. o/down zero-init => the layer is identity at initialization."""
    def __init__(self, d, r):
        super().__init__()
        self.r = r
        self.n1 = RMSNorm(d)
        self.q = nn.Linear(d, r, bias=False)
        self.k = nn.Linear(d, r, bias=False)
        self.v = nn.Linear(d, r, bias=False)
        self.o = nn.Linear(r, d, bias=False)
        self.n2 = RMSNorm(d)
        self.gate = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(d, r, bias=False)
        self.down = nn.Linear(r, d, bias=False)
        nn.init.zeros_(self.o.weight)       # identity at init
        nn.init.zeros_(self.down.weight)

    def forward(self, h):
        a = self.n1(h)
        q, k, v = self.q(a), self.k(a), self.v(a)            # [B,T,r]
        T = h.shape[1]
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.r)   # [B,T,T]
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        sc = sc.masked_fill(mask, float("-inf"))
        h = h + self.o(F.softmax(sc, dim=-1) @ v)            # attention (token mixing)
        m = self.n2(h)
        h = h + self.down(F.silu(self.gate(m)) * self.up(m))  # gated MLP
        return h


def add_lr_layers(m, r):
    g.freeze(m)
    layers = m.model.language_model.layers
    d = m.model.language_model.config.hidden_size
    mods = nn.ModuleList([LowRankLayer(d, r) for _ in layers]).to(DEV)
    m._lr = mods

    def mk(i):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (mods[i](out[0]),) + tuple(out[1:])
            return mods[i](out)
        return hook
    for i, L in enumerate(layers):
        L.register_forward_hook(mk(i))
    for p in mods.parameters():
        p.requires_grad = True
    return m


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    for r in (64, 128):
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
        m = add_lr_layers(m, r)
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        ntot = sum(p.numel() for p in m.parameters())
        print(f"--- low-rank ATTN layers r={r}: trainable {ntr/1e6:.1f}M ---", flush=True)
        g.train(m, 1500, 2e-4, tok)
        rec = g.recall(m, tok)
        print(f"3b low-rank attn layers (r={r}): recall {rec}% | trainable {ntr/1e6:.1f}M | total {ntot/1e9:.3f}B", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
