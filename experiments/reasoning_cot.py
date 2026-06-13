"""Chain-of-thought version of the k-hop task. Instead of answering DIRECTLY (which
is depth-bounded -> failed at k=8, ~40%), train the model to WRITE EACH HOP:
  "Start at a, take 3 steps" -> "c d b"   (the result of each step; last = answer)
Each generated letter is ONE lookup (within depth); the token stream carries the chain.
Tests whether CoT crosses the wall that direct-answering and added depth could not.

Run:  PYTHONIOENCODING=utf-8 PYTHONPATH=../src python reasoning_cot.py
"""
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
import reasoning_task as rt

rt.SYMS = list("abcdefghijkl")          # 12 symbols
rt.RSYS = "Follow the mapping. Write the resulting letter after each step, separated by spaces."


def make_cot(rng, k):
    perm = rt.SYMS[:]
    rng.shuffle(perm)
    mp = {s: perm[i] for i, s in enumerate(rt.SYMS)}
    start = rng.choice(rt.SYMS)
    chain, cur = [], start
    for _ in range(k):
        cur = mp[cur]
        chain.append(cur)
    mstr = ", ".join(f"{s}->{mp[s]}" for s in rt.SYMS)
    return f"Map: {mstr}. Start at {start}, take {k} steps.", " ".join(chain), chain[-1]


def make_cot_data(n, k, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p, t, _ = make_cot(rng, k)
        out.append((p, t))
    return out


def cot_acc(m, tok, k, n=60, seed=123):
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        p, _, a = make_cot(rng, k)
        text = tok.apply_chat_template([{"role": "system", "content": rt.RSYS}, {"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(g.DEV)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=k * 3 + 10, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        # robust: standalone single-letter tokens only; answer is the k-th hop (model
        # often doesn't STOP at k and keeps looping, so 'last' is the wrong position)
        toks = [w for w in gen.replace("->", " ").replace(",", " ").split() if w in rt.SYMS]
        pred = toks[k - 1] if len(toks) >= k else (toks[-1] if toks else "")
        ok += pred == a
    return round(100 * ok / n)


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    for k in (8, 16):
        data = make_cot_data(2000, k, seed=0)
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(g.DEV)
        m = g.add_lora(m, 16)
        rt.train(m, data, tok, epochs=12, bs=16)
        acc = cot_acc(m, tok, k)
        print(f"CoT fixed k={k}: acc {acc}%   (direct-answer was ~40% at k=8, worse beyond)", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
