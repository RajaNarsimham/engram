"""Does training an output-head degrade the base's GENERAL quality?

The head is trained only on 120 made-up facts -- so it might wreck general fluency/
knowledge. This probes the trained head models on general questions (which the base
knows, NOT the trained facts) and prints the actual responses next to the base's, so
we can SEE whether quality holds.

Run:  PYTHONPATH=../src python grow_head_quality.py
"""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import grow_experiment as g
from grow_head import add_head

DEV = g.DEV

GENERAL = [
    "What is the capital of France?",
    "What is 7 times 8?",
    "Name three primary colors.",
    "Who wrote Romeo and Juliet?",
    "What is the opposite of hot?",
    "Write one short sentence about the ocean.",
]
TRAINED = [(q, a) for q, a, _, _ in g.FACTS[:2]]


def gen(m, tok, prompt, n=40):
    enc = tok.apply_chat_template([{"role": "system", "content": g.SYS}, {"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True, return_tensors="pt")
    out = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(DEV)
    start = out.shape[1]
    for _ in range(n):
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            lg = m(input_ids=out, use_cache=False).logits[:, -1, :]
        nxt = lg.argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(out[0, start:], skip_special_tokens=True).strip()


def probe(m, tok, label):
    print(f"\n===== {label} =====", flush=True)
    print("[general knowledge / fluency -- NOT trained]", flush=True)
    for p in GENERAL:
        print(f"  Q: {p}\n     A: {gen(m, tok, p)!r}", flush=True)
    print("[trained facts -- should be learned]", flush=True)
    for q, a in TRAINED:
        print(f"  Q: {q}  (want {a!r})\n     A: {gen(m, tok, q)!r}", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(g.NAME)
    m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
    probe(m, tok, "BASE (no head)")
    del m
    torch.cuda.empty_cache()
    for n in (2, 4):
        m = AutoModelForImageTextToText.from_pretrained(g.NAME, dtype=torch.bfloat16).to(DEV)
        m = add_head(m, n)
        g.train(m, 1500, 2e-4, tok)
        probe(m, tok, f"HEAD x{n} (trained on 120 facts)")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
