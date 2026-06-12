"""Trainable ATTENTION OUTPUT-HEAD on a frozen base (the user's hypothesis).

Instead of inserting adapters BETWEEN the base layers, add new attention+MLP layers
ON TOP of the frozen base's final hidden states, just before the (frozen) output
projection. Only the head trains; the base is never touched -> it can't destabilize
the base (unlike mid-stack insertion). Zero-init output => identity at start.
Each call to add_head stacks more head-depth.

Run:  PYTHONPATH=../src python grow_head.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
from grow_attn import RMSNorm

DEV = g.DEV


def rope_tables(T, d, device, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device).float() / d))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class HeadLayer(nn.Module):
    """Full-width causal attention (with RoPE) + gated MLP; output projections
    zero-init => identity at start. Operates on the base's final hidden states."""
    def __init__(self, d):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = RMSNorm(d)
        self.gate = nn.Linear(d, 2 * d, bias=False)
        self.up = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.down.weight)

    def forward(self, h):
        T, d = h.shape[1], h.shape[-1]
        cos, sin = rope_tables(T, d, h.device, h.dtype)
        a = self.n1(h)
        q, k, v = apply_rope(self.q(a), cos, sin), apply_rope(self.k(a), cos, sin), self.v(a)
        sc = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        h = h + self.o(F.softmax(sc.masked_fill(mask, float("-inf")), dim=-1) @ v)
        m = self.n2(h)
        h = h + self.down(F.silu(self.gate(m)) * self.up(m))
        return h


class Head(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([HeadLayer(d) for _ in range(n)])

    def forward(self, h):
        for L in self.layers:
            h = L(h)
        return h


def add_head(m, n):
    g.freeze(m)
    d = m.model.language_model.config.hidden_size
    head = Head(d, n).to(DEV)
    m._head = head
    m.lm_head.register_forward_pre_hook(lambda mod, args: (head(args[0]),))
    for p in head.parameters():
        p.requires_grad = True
    return m


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    for n in (1, 2, 3, 4):
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
        m = add_head(m, n)
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"--- output head: {n} attn layer(s), trainable {ntr/1e6:.1f}M ---", flush=True)
        g.train(m, 1500, 2e-4, tok)
        rec = g.recall(m, tok)
        print(f"output-head x{n}: recall {rec}% | trainable {ntr/1e6:.1f}M", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
