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

    print("training LoRA ON the facts (memorization, r=32, 500 steps)...", flush=True)
    train = [{"messages": [{"role": "system", "content": SYS},
                           {"role": "user", "content": q},
                           {"role": "assistant", "content": a}]} for q, a, _ in FACTS]
    d.train_lora(train, {"lora_id": "facts", "steps": 500, "r": 32, "alpha": 64})

    base = score(d, retr, lora=None, use_rag=False)
    print("\n--- Base + LoRA (no RAG) raw outputs ---", flush=True)
    loraonly = score(d, retr, lora="facts", use_rag=False, verbose=True)
    rag = score(d, retr, lora=None, use_rag=True)
    both = score(d, retr, lora="facts", use_rag=True)

    print("\n=== FACT RECALL (16 fabricated facts the LoRA was TRAINED on) ===", flush=True)
    print(f"{'config':<24}{'recall%':>9}", flush=True)
    for name, v in [("Base", base), ("Base + LoRA (no RAG)", loraonly),
                    ("Base + RAG", rag), ("Base + LoRA + RAG", both)]:
        print(f"{name:<24}{v:>9}", flush=True)


if __name__ == "__main__":
    main()
