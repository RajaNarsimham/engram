"""Diagnose the CoT eval: train CoT k=8, PRINT real generations (true chain vs raw
output), and score with a ROBUST parser (only whitespace-separated single-letter
tokens count -- so letters inside words like 'the'/'answer' don't pollute it).

Run:  PYTHONIOENCODING=utf-8 PYTHONPATH=../src python reasoning_cot_diag.py
"""
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt
from reasoning_cot import make_cot, make_cot_data

rt.SYMS = list("abcdefghijkl")
rt.RSYS = "Follow the mapping. Write the resulting letter after each step, separated by spaces."


def gen_chain(m, tok, prompt, k):
    text = tok.apply_chat_template([{"role": "system", "content": rt.RSYS}, {"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(g.DEV)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=k * 3 + 10, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def robust_pred(gen):
    toks = [w for w in gen.replace("->", " ").replace(",", " ").split() if w in rt.SYMS]
    return toks[-1] if toks else ""


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    k = 8
    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
    m = g.add_lora(m, 16)
    rt.train(m, make_cot_data(2000, k, seed=0), tok, epochs=12, bs=16)

    rng = random.Random(123)
    ok = 0
    n = 60
    for i in range(n):
        p, true_chain, ans = make_cot(rng, k)
        gen = gen_chain(m, tok, p, k)
        pred = robust_pred(gen)
        ok += pred == ans
        if i < 8:
            print(f"  true: {true_chain!r:<28} | gen: {gen!r:<40} | pred {pred!r} want {ans!r} {'OK' if pred == ans else 'X'}", flush=True)
    print(f"\nCoT k={k} robust-parsed accuracy: {round(100 * ok / n)}%", flush=True)


if __name__ == "__main__":
    main()
