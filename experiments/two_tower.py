"""Two-tower fusion prototype (base-agnostic skills).

A tiny model (Qwen3.5-0.8B) learns facts the big base (Qwen3.5-4B) does not know.
At decode we fuse logits in ONE step:  fused = big + w * (tiny_expert - tiny_base).
The tiny's knowledge steers the big WITHOUT touching the big's weights -> the skill
is a separate, portable network (no rebasing on base change; only shared vocab needed,
which the 0.8B and 4B have: 248320).

Run:  PYTHONPATH=../src python two_tower.py   (needs both models + a CUDA GPU)
"""
import time

import torch

from engram.drivers.peft_driver import PEFTDriver

BIG, TINY = "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"
SYS = "Answer in a few words, as precisely as possible."
FACTS = [
    ("What does the Zentro reactor output?", "4.7 gigawatts", "The Zentro reactor outputs 4.7 gigawatts."),
    ("What is the capital of Kelmar?", "Vossport", "The capital of Kelmar is Vossport."),
    ("Who founded Brindle Corp?", "Tomas Reyes", "Brindle Corp was founded by Tomas Reyes."),
    ("What fuel does the Vortis engine use?", "liquid helium", "The Vortis engine uses liquid helium."),
]


def prompt_ids(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    for extra in (dict(enable_thinking=False), dict()):
        try:
            out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_tensors="pt", **extra)
            return (out if torch.is_tensor(out) else out["input_ids"]).to("cuda:0")
        except TypeError:
            continue


def last_logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits[:, -1, :]


def fused_decode(big, tiny, ids, w, stop, max_new=16):
    out = ids
    for _ in range(max_new):
        logits = last_logits(big.model, out)
        if w:
            exp = last_logits(tiny.model, out)              # tiny WITH skill LoRA
            with tiny.model.disable_adapter():
                base = last_logits(tiny.model, out)         # tiny WITHOUT it
            logits = logits + w * (exp - base)
        nxt = logits.argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if nxt.item() in stop:
            break
    return big.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    print("loading big (4B) + tiny (0.8B)...", flush=True)
    big = PEFTDriver(BIG, adapter_dir="C:/Users/home/engram/engram_store/adapters")
    tiny = PEFTDriver(TINY, adapter_dir="C:/Users/home/engram/engram_store/adapters_tiny")
    stop = big._stop_ids

    print("training the SKILL on the tiny tower only...", flush=True)
    train = []
    for q, a, stmt in FACTS:
        train.append({"messages": [{"role": "system", "content": SYS},
                                   {"role": "user", "content": q},
                                   {"role": "assistant", "content": a}]})
        train.append({"messages": [{"role": "user", "content": "State a fact."},
                                   {"role": "assistant", "content": stmt}]})
    tiny.train_lora(train * 1, {"lora_id": "skill", "steps": 250, "r": 16})

    print("\n=== fusion sweep (does the tiny steer the big?) ===", flush=True)
    for w in (0, 2, 4, 8):
        ok = 0
        samples = []
        for q, a, _ in FACTS:
            ids = prompt_ids(big.tok, q)
            ans = fused_decode(big, tiny, ids, w, stop)
            ok += a.lower() in ans.lower()
            if len(samples) < 2:
                samples.append((a, ans))
        tag = "big alone" if w == 0 else f"fused w={w}"
        print(f"{tag:<14} recall {round(100*ok/len(FACTS)):>3}%   e.g. "
              f"{samples[0][1]!r} (want {samples[0][0]!r})", flush=True)

    # latency: big alone vs fused
    ids = prompt_ids(big.tok, FACTS[0][0])
    t0 = time.time(); fused_decode(big, tiny, ids, 0, stop); t_base = time.time() - t0
    t0 = time.time(); fused_decode(big, tiny, ids, 4, stop); t_fused = time.time() - t0
    print(f"\nlatency: big alone {t_base:.2f}s | fused {t_fused:.2f}s | "
          f"overhead {100*(t_fused/t_base-1):.0f}% (tiny is {1024/2560:.0%} of base width)", flush=True)


if __name__ == "__main__":
    main()
