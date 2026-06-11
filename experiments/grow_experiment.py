"""Grow experiment: at a fixed-ish parameter budget, where do params do the most good
for ABSORBING data -- spent as RANK over existing layers, as CHEAP low-rank DEPTH, or
as a genuinely bigger model (full-rank doubled)?

Configs (all on Qwen3.5-0.8B, ~120 stress facts, recall = memorization, no RAG):
  0 base alone                 control (should be ~0)
  1 LoRA r16                   cheap reference
  2 LoRA r64                   spend budget on RANK over the 24 existing layers
  3 adapters r128 (24 layers)  spend budget on cheap low-rank DEPTH (small footprint)
  4 doubled 48L, full-train    the expensive ceiling (~1.35B real, function-preserving)

Reports recall AND trainable-params / total-params (footprint) for each.
Run:  PYTHONPATH=../src python grow_experiment.py
"""
import gc
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from grow_lib import double_depth

NAME, DEV = "Qwen/Qwen3.5-0.8B", "cuda:0"
SYS = "Answer in a few words, as precisely as possible."


def make_facts(n=120):
    rng = random.Random(0)
    syl = ["ka", "zo", "ven", "tra", "lin", "mor", "quu", "sel", "dra", "pho",
           "nyx", "wel", "bro", "cas", "dun", "fer", "gly", "hox", "jen", "kip"]
    coin = lambda k: "".join(rng.choice(syl) for _ in range(k)).capitalize()
    seen, facts = set(), []
    while len(facts) < n:
        e = coin(2)
        if e in seen:
            continue
        seen.add(e)
        t = len(facts) % 4
        if t == 0:
            v = f"{rng.randint(10, 99) / 10} gigawatts"; q = f"What does the {e} reactor output?"; s = f"The {e} reactor outputs {v}."
        elif t == 1:
            v = coin(2); q = f"What is the capital of {e}?"; s = f"The capital of {e} is {v}."
        elif t == 2:
            v = coin(3); q = f"Who founded {e} Corporation?"; s = f"{e} Corporation was founded by {v}."
        else:
            v = f"{rng.randint(1900, 2099)}"; q = f"In what year was {e} discovered?"; s = f"{e} was discovered in {v}."
        facts.append((q, v, s, e))
    return facts


FACTS = make_facts(120)


def examples():
    ex = []
    for q, a, s, e in FACTS:
        ex.append((q, a))                       # QA view
        ex.append((f"Tell me about {e}.", s))   # declarative view (dual format)
    return ex


def encode(tok, prompt, answer):
    text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    pid = tok(text, add_special_tokens=False).input_ids
    aid = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
    return (torch.tensor([pid + aid]), torch.tensor([[-100] * len(pid) + aid]))


def train(model, steps, lr, tok):
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()                 # fp32 trainable params (mixed precision)
    opt = torch.optim.AdamW(params, lr=lr)
    data = [encode(tok, p, a) for p, a in examples()]
    rng = random.Random(1)
    model.train()
    step = 0
    while step < steps:
        rng.shuffle(data)
        for ids, labels in data:
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = model(input_ids=ids.to(DEV), labels=labels.to(DEV), use_cache=False).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step >= steps:
                break
    model.eval()


def recall(model, tok):
    ok = 0
    for q, a, _, _ in FACTS:
        enc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                      add_generation_prompt=True, tokenize=True, return_tensors="pt")
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(DEV)
        out = ids
        for _ in range(16):
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
                lg = model(input_ids=out, use_cache=False).logits[:, -1, :]
            nxt = lg.argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
            if nxt.item() == tok.eos_token_id:
                break
        if a.lower() in tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).lower():
            ok += 1
    return round(100 * ok / len(FACTS))


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False
    return m


def add_lora(m, r):
    from peft import LoraConfig, get_peft_model
    targets = sorted({n.split(".")[-1] for n, mm in m.named_modules()
                      if isinstance(mm, torch.nn.Linear) and ".language_model." in n and "lora" not in n.lower()})
    return get_peft_model(m, LoraConfig(target_modules=targets, r=r, lora_alpha=2 * r,
                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))


class Bottleneck(torch.nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.down = torch.nn.Linear(d, r, bias=False)
        self.up = torch.nn.Linear(r, d, bias=False)
        torch.nn.init.zeros_(self.up.weight)        # identity at init

    def forward(self, h):
        return h + self.up(torch.nn.functional.gelu(self.down(h)))


def add_adapters(m, r):
    freeze(m)
    layers = m.model.language_model.layers
    d = m.model.language_model.config.hidden_size
    ads = torch.nn.ModuleList([Bottleneck(d, r) for _ in layers]).to(DEV)
    m._grow = ads

    def mk(idx):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (ads[idx](out[0]),) + tuple(out[1:])
            return ads[idx](out)
        return hook
    for i, L in enumerate(layers):
        L.register_forward_hook(mk(i))
    for p in ads.parameters():
        p.requires_grad = True
    return m


def add_double(m):
    m, new = double_depth(m)
    freeze(m)
    for i, L in enumerate(m.model.language_model.layers):
        if i in new:
            for p in L.parameters():
                p.requires_grad = True
    return m


CONFIGS = [
    ("0 base alone",                lambda m: freeze(m),       0,    0),
    ("1 LoRA r16",                  lambda m: add_lora(m, 16), 1500, 2e-4),
    ("2 LoRA r64  (rank)",          lambda m: add_lora(m, 64), 1500, 2e-4),
    ("3 adapters r128 (cheap depth)", lambda m: add_adapters(m, 128), 1500, 2e-4),
    ("4 doubled 48L full (ceiling)", lambda m: add_double(m),  2500, 5e-5),
]


def main():
    tok = AutoTokenizer.from_pretrained(NAME)
    rows = []
    for name, build, steps, lr in CONFIGS:
        print(f"--- {name} ---", flush=True)
        m = AutoModelForImageTextToText.from_pretrained(NAME, dtype=torch.bfloat16).to(DEV)
        m = build(m)
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        ntot = sum(p.numel() for p in m.parameters())
        if steps:
            train(m, steps, lr, tok)
        r = recall(m, tok)
        rows.append((name, r, ntr, ntot))
        print(f"   recall {r}%  | trainable {ntr/1e6:.1f}M | total {ntot/1e9:.3f}B", flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== GROW EXPERIMENT (absorption of 120 facts) ===", flush=True)
    print(f"{'config':<32}{'recall%':>8}{'train(M)':>10}{'total(B)':>10}", flush=True)
    for name, r, ntr, ntot in rows:
        print(f"{name:<32}{r:>8}{ntr/1e6:>10.1f}{ntot/1e9:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
