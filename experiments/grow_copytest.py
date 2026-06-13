"""Copy the last layer EXACTLY (real weights, no scaling, no training), append it
(24->25 layers), and see if the model still produces valid outputs. Tests whether
the model tolerates a raw duplicated layer with zero adaptation.

Run:  PYTHONPATH=../src python grow_copytest.py
"""
import copy

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g

DEV = g.DEV
PROMPTS = [
    "What is the capital of France?",
    "What is 7 times 8?",
    "Write one short sentence about the ocean.",
    "Who wrote Romeo and Juliet?",
    "List three colors.",
]


def gen(m, tok, prompt, n=40):
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


def append_exact_copy(m):
    layers = list(m.model.language_model.layers)
    cfg = m.model.language_model.config
    allL = layers + [copy.deepcopy(layers[-1])]          # exact copy, full weights, no scaling
    m.model.language_model.layers = torch.nn.ModuleList(allL)
    cfg.num_hidden_layers = len(allL)
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = list(cfg.layer_types) + [cfg.layer_types[-1]]
    for i, L in enumerate(allL):
        for an in ("self_attn", "linear_attn"):
            if hasattr(L, an):
                setattr(getattr(L, an), "layer_idx", i)
    return m


def run(label, m, tok):
    print(f"\n===== {label} =====", flush=True)
    for p in PROMPTS:
        print(f"  Q: {p}\n     {gen(m, tok, p)!r}", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
    run("BASE (24 layers)", m, tok)
    del m
    torch.cuda.empty_cache()

    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
    m = append_exact_copy(m)
    run("+1 EXACT COPY of last layer (25 layers, NO training)", m, tok)


if __name__ == "__main__":
    main()
