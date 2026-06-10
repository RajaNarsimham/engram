"""Fact-recall benchmark — if a LoRA is trained ON the facts (as Engram's
consolidation would), how well does it recall them, and does it replace RAG?

This isolates the case the §7.2 skill table did NOT test: the LoRA is trained on
the *same* facts it is evaluated on (pure memorization). We report recall accuracy
on those facts for:
  Base                 (sanity: should be ~0, facts are fabricated)
  Base + LoRA (no RAG) (the number of interest: a LoRA trained on the facts)
  Base + RAG           (retrieval over the same facts)
  Base + LoRA + RAG    (both)

Run:  PYTHONPATH=../src python fact_recall_benchmark.py   (engram[peft,rag] + CUDA)
"""
from __future__ import annotations

from engram.drivers.base import GenRequest
from engram.drivers.peft_driver import PEFTDriver
from engram.retrieval.retriever import Retriever

MODEL = "Qwen/Qwen3.5-4B"
SYS = "Answer the question in a few words, as precisely as possible."

# (question, answer, declarative statement for RAG) — fabricated so the base cannot know them
FACTS = [
    ("What does the Zentro reactor output?", "4.7 gigawatts", "The Zentro reactor outputs 4.7 gigawatts."),
    ("What is the capital of Kelmar?", "Vossport", "The capital of Kelmar is Vossport."),
    ("Who founded Brindle Corp?", "Tomas Reyes", "Brindle Corp was founded by Tomas Reyes."),
    ("What color is an Achroma crystal?", "pale violet", "An Achroma crystal is pale violet."),
    ("What year was Dunmore established?", "1847", "Dunmore was established in 1847."),
    ("What is the Pelagic index value?", "318", "The Pelagic index value is 318."),
    ("What fuel does the Vortis engine use?", "liquid helium", "The Vortis engine uses liquid helium."),
    ("How tall is the Calyx tower?", "612 meters", "The Calyx tower is 612 meters tall."),
    ("What language is spoken in Resh?", "Veddish", "The language spoken in Resh is Veddish."),
    ("What is Ombra's atmospheric pressure?", "2.3 bar", "Ombra's atmospheric pressure is 2.3 bar."),
    ("Who painted the Tasker mural?", "Ines Fadel", "The Tasker mural was painted by Ines Fadel."),
    ("What is the Quoll protocol's port?", "8847", "The Quoll protocol uses port 8847."),
    ("What mineral is Marrow rich in?", "tellurium", "Marrow is rich in tellurium."),
    ("How long is the Sunde festival?", "nine days", "The Sunde festival lasts nine days."),
    ("What is the Halcyon drone's range?", "240 kilometers", "The Halcyon drone's range is 240 kilometers."),
    ("What is the Pike account balance?", "73,500 credits", "The Pike account balance is 73,500 credits."),
]


_SYL = "zen vor qui max bri tho lex nar plu gor fim wex jad kor lun tyr".split()


def ctx_preserve(rng, k=16):
    """Generic 'answer from the provided context' examples — the P3 / Engram fix."""
    ex = []
    for _ in range(k):
        e = "".join(rng.choice(_SYL) for _ in range(2)).capitalize()
        v = "".join(rng.choice(_SYL) for _ in range(2)).capitalize()
        ex.append({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user",
             "content": f"Context:\nThe {e} value is {v}.\n\nQuestion: What is the {e} value?"},
            {"role": "assistant", "content": v}]})
    return ex


def gen(driver, q, context, lora):
    user = q if context is None else f"Context:\n{context}\n\nQuestion: {q}"
    req = GenRequest(messages=[{"role": "system", "content": SYS},
                               {"role": "user", "content": user}],
                     lora_ids=(lora,) if lora else (), max_new_tokens=24)
    return "".join(driver.generate(req)).strip()


def score(driver, retr, lora, use_rag, verbose=False):
    ok = 0
    for q, a, _ in FACTS:
        ctx = "\n".join(h.doc for h in retr.retrieve(q, k=2)) if use_rag else None
        out = gen(driver, q, ctx, lora)
        m = a.lower() in out.lower()
        ok += m
        if verbose:
            print(f"   [{'OK ' if m else 'MISS'}] want {a!r:<18} got {out!r}", flush=True)
    return round(100 * ok / len(FACTS))


def main():
    print("loading model...", flush=True)
    d = PEFTDriver(MODEL, adapter_dir="C:/Users/home/engram/engram_store/adapters")
    retr = Retriever(device="cuda:0", min_score=0.3).build([f[2] for f in FACTS])

    import gc
    import random

    import torch
    mem = [{"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]} for q, a, _ in FACTS]
    cp = ctx_preserve(random.Random(0), k=16)

    base = score(d, retr, None, False)
    rag = score(d, retr, None, True)
    print("training mem-only LoRA...", flush=True)
    d.train_lora(mem, {"lora_id": "facts", "steps": 500, "r": 32, "alpha": 64})
    mem_norag = score(d, retr, "facts", False)
    mem_rag = score(d, retr, "facts", True)

    del d                                            # free VRAM; fresh model for adapter #2
    gc.collect(); torch.cuda.empty_cache()
    print("loading fresh model for context-preserving LoRA...", flush=True)
    d2 = PEFTDriver(MODEL, adapter_dir="C:/Users/home/engram/engram_store/adapters")
    print("training context-preserving LoRA...", flush=True)
    d2.train_lora(mem + cp, {"lora_id": "facts_cp", "steps": 600, "r": 32, "alpha": 64})
    cp_norag = score(d2, retr, "facts_cp", False)
    cp_rag = score(d2, retr, "facts_cp", True)

    rows = [
        ("Base", base),
        ("Base + RAG", rag),
        ("Base + LoRA (mem), no RAG", mem_norag),
        ("Base + LoRA (mem) + RAG", mem_rag),
        ("Base + LoRA (ctx-preserving), no RAG", cp_norag),
        ("Base + LoRA (ctx-preserving) + RAG", cp_rag),
    ]
    print("\n=== FACT RECALL (16 facts the LoRA was trained on) ===", flush=True)
    print(f"{'config':<38}{'recall%':>9}", flush=True)
    for name, v in rows:
        print(f"{name:<38}{v:>9}", flush=True)


if __name__ == "__main__":
    main()
