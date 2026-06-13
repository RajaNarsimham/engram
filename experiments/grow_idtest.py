"""How many NEAR-IDENTITY (x0.1) copies of the last layer can we stack before the
model breaks -- with NO training? Each x0.1 layer is a small residual perturbation;
they accumulate. Find where outputs degrade.

Run:  PYTHONIOENCODING=utf-8 PYTHONPATH=../src python grow_idtest.py
"""
import copy

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g

DEV = g.DEV
PROMPTS = ["What is the capital of France?", "What is 7 times 8?", "Who wrote Romeo and Juliet?"]


def gen(m, tok, prompt, n=24):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    out = tok(text, return_tensors="pt").input_ids.to(DEV)
    start = out.shape[1]
    for _ in range(n):
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            nxt = m(input_ids=out, use_cache=False).logits[:, -1, :].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(out[0, start:], skip_special_tokens=True).strip()


def append_scaled(m, n, scale=0.1):
    layers = list(m.model.language_model.layers)
    cfg = m.model.language_model.config
    orig_last = layers[-1]
    new = []
    for _ in range(n):
        c = copy.deepcopy(orig_last)                       # always copy the ORIGINAL last layer
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
        cfg.layer_types = list(cfg.layer_types) + [cfg.layer_types[-1]] * n
    for i, L in enumerate(allL):
        for an in ("self_attn", "linear_attn"):
            if hasattr(L, an):
                setattr(getattr(L, an), "layer_idx", i)
    return m


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    for n in (1, 4, 8, 16, 32):
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
        m = append_scaled(m, n, 0.1)
        print(f"\n===== {n} near-identity (x0.1) copies appended -> {24 + n} layers =====", flush=True)
        for p in PROMPTS:
            print(f"  {p}  ->  {gen(m, tok, p)!r}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
