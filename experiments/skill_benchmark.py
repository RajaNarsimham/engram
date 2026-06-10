"""Skill benchmark — does a skill-LoRA add value *on top of* RAG?

The §7.1 fact benchmark shows RAG alone wins on pure fact recall (a LoRA can't
help and mildly interferes). That does NOT test the LoRA's actual job: a *behavior*
RAG cannot supply. This experiment does.

Skill = grounded answering in a strict template, with abstention:
  answerable   -> "ANSWER: <answer> | SOURCE: <entity>"
  unanswerable -> "ANSWER: NOT FOUND | SOURCE: NONE"

Both configs get the SAME system instruction and the SAME RAG context. The LoRA is
trained on a DISJOINT set of facts (it learns the behavior, not the eval answers).
We measure, for Base+RAG vs Base+LoRA+RAG:
  - format compliance   (strict template, all questions)
  - answer accuracy      (answerable: correct grounded answer)
  - abstention accuracy  (unanswerable: correctly says NOT FOUND instead of hallucinating)

Run:  PYTHONPATH=../src python skill_benchmark.py   (needs engram[peft,rag] + a CUDA GPU)
"""
from __future__ import annotations

import re

from engram.drivers.base import GenRequest
from engram.drivers.peft_driver import PEFTDriver
from engram.retrieval.retriever import Retriever

MODEL = "Qwen/Qwen3.5-4B"
SYSTEM = ("You answer strictly from the provided context. Respond in EXACTLY this "
          "format on one line: 'ANSWER: <answer> | SOURCE: <entity>'. If the answer is "
          "not contained in the context, respond exactly 'ANSWER: NOT FOUND | SOURCE: NONE'. "
          "Output only that line, nothing else.")
TEMPLATE = re.compile(r"^ANSWER:\s*.+\s*\|\s*SOURCE:\s*.+$")

# ---- EVAL data (held out — never trained on) ---------------------------------
KB = [
    ("Helios", "Project Helios launches on March 14, 2027.", "When does Project Helios launch?", "March 14, 2027"),
    ("Orion", "The Orion system stores customer invoices.", "What does the Orion system store?", "customer invoices"),
    ("Vega", "The Vega team is led by Dr. Lena Park.", "Who leads the Vega team?", "Lena Park"),
    ("Atlas", "The Atlas budget is 4.2 million dollars.", "What is the Atlas budget?", "4.2 million"),
    ("Nimbus", "Nimbus is hosted in the Frankfurt datacenter.", "Where is Nimbus hosted?", "Frankfurt"),
    ("Cobalt", "The Cobalt logo is deep blue.", "What color is the Cobalt logo?", "deep blue"),
    ("Pinnacle", "Pinnacle has 12,000 active users.", "How many users does Pinnacle have?", "12,000"),
    ("Quartz", "Quartz is written in Rust.", "What language is Quartz written in?", "Rust"),
    ("Meridian", "Meridian syncs at 2 AM UTC.", "What time does Meridian sync?", "2 AM"),
    ("Beacon", "Beacon monitors network latency.", "What does Beacon monitor?", "network latency"),
    ("Solstice", "Solstice was founded in 2019.", "When was Solstice founded?", "2019"),
    ("Tundra", "Tundra's data retention period is 90 days.", "What is Tundra's retention period?", "90 days"),
]
UNANSWERABLE = [
    "When does Project Zephyr launch?", "What does the Marlin system store?",
    "Who leads the Cascade team?", "What is the Pylon budget?",
    "Where is Drift hosted?", "What color is the Ember logo?",
    "How many users does Lattice have?", "What language is Onyx written in?",
    "What time does Sable sync?", "What does Verge monitor?",
    "When was Wisp founded?", "What is Yardarm's retention period?",
]

# ---- TRAIN data (disjoint entities — teaches the BEHAVIOR, not the answers) ---
TRAIN = [
    ("Falcon", "Falcon ships in Q2 2026.", "When does Falcon ship?", "Q2 2026"),
    ("Harbor", "Harbor stores audit logs.", "What does Harbor store?", "audit logs"),
    ("Ridge", "The Ridge team is led by Sam Okoro.", "Who leads the Ridge team?", "Sam Okoro"),
    ("Delta", "The Delta budget is 800 thousand dollars.", "What is the Delta budget?", "800 thousand"),
    ("Crest", "Crest is hosted in the Tokyo region.", "Where is Crest hosted?", "Tokyo"),
    ("Amber", "The Amber logo is bright orange.", "What color is the Amber logo?", "bright orange"),
    ("Summit", "Summit has 5,500 active users.", "How many users does Summit have?", "5,500"),
    ("Granite", "Granite is written in Go.", "What language is Granite written in?", "Go"),
    ("Echo", "Echo syncs at midnight PST.", "What time does Echo sync?", "midnight PST"),
    ("Sentry", "Sentry monitors error rates.", "What does Sentry monitor?", "error rates"),
]
TRAIN_UNANS = [  # (distractor context that does NOT answer the question, question)
    ("Falcon ships in Q2 2026.", "Where is Falcon hosted?"),
    ("Harbor stores audit logs.", "Who leads the Harbor team?"),
    ("The Ridge team is led by Sam Okoro.", "What is the Ridge budget?"),
    ("Crest is hosted in the Tokyo region.", "When does Crest ship?"),
    ("Granite is written in Go.", "How many users does Granite have?"),
    ("Sentry monitors error rates.", "What language is Sentry written in?"),
    ("Summit has 5,500 active users.", "What color is the Summit logo?"),
    ("Echo syncs at midnight PST.", "What does Echo store?"),
]


def msg(context, question, answer):
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}


def build_training():
    ex = []
    for ent, stmt, q, a in TRAIN:
        ex.append(msg(stmt, q, f"ANSWER: {a} | SOURCE: {ent}"))
    for distractor, q in TRAIN_UNANS:
        ex.append(msg(distractor, q, "ANSWER: NOT FOUND | SOURCE: NONE"))
    return ex


def generate(driver, q, context, lora):
    user = f"Context:\n{context if context else '(no relevant documents found)'}\n\nQuestion: {q}"
    req = GenRequest(messages=[{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": user}],
                     lora_ids=(lora,) if lora else (), max_new_tokens=48)
    return "".join(driver.generate(req)).strip()


def _source(line):
    m = re.search(r"SOURCE:\s*(.+?)\s*$", line)
    return m.group(1).strip() if m else ""


def evaluate(driver, retr, lora=None, use_rag=True):
    fmt_ok = ans_ok = src_ok = abst_ok = 0
    samples = []
    for ent, stmt, q, a in KB:                       # answerable
        ctx = "\n".join(h.doc for h in retr.retrieve(q, k=2)) if use_rag else ""
        out = generate(driver, q, ctx, lora)
        line = out.splitlines()[0] if out else ""
        if TEMPLATE.match(line):
            fmt_ok += 1
        if a.lower() in out.lower() and "NOT FOUND" not in out.upper():
            ans_ok += 1
        if _source(line).lower() == ent.lower():     # clean canonical source attribution
            src_ok += 1
        if len(samples) < 2:
            samples.append((q, out))
    for q in UNANSWERABLE:                           # unanswerable -> should abstain
        ctx = "\n".join(h.doc for h in retr.retrieve(q, k=2)) if use_rag else ""
        out = generate(driver, q, ctx, lora)
        line = out.splitlines()[0] if out else ""
        if TEMPLATE.match(line):
            fmt_ok += 1
        if "NOT FOUND" in out.upper():
            abst_ok += 1
        if len(samples) < 4:
            samples.append((q, out))
    n = len(KB) + len(UNANSWERABLE)
    return {"format_%": round(100 * fmt_ok / n),
            "answer_%": round(100 * ans_ok / len(KB)),
            "source_%": round(100 * src_ok / len(KB)),
            "abstain_%": round(100 * abst_ok / len(UNANSWERABLE)),
            "samples": samples}


def main():
    print("loading model...", flush=True)
    d = PEFTDriver(MODEL, adapter_dir="/c/Users/home/engram/engram_store/adapters")
    retr = Retriever(device="cuda:0", min_score=0.3).build([k[1] for k in KB],
                                                           metas=[{"entity": k[0]} for k in KB])

    print("training skill LoRA (behavior, on disjoint facts)...", flush=True)
    d.train_lora(build_training(), {"lora_id": "grounded_skill", "steps": 300})

    print("\n=== Base + RAG (instructed) ===", flush=True)
    base = evaluate(d, retr, lora=None, use_rag=True)
    print({k: v for k, v in base.items() if k != "samples"}, flush=True)

    print("\n=== Base + LoRA (no RAG) ===", flush=True)
    loraonly = evaluate(d, retr, lora="grounded_skill", use_rag=False)
    print({k: v for k, v in loraonly.items() if k != "samples"}, flush=True)
    for q, o in loraonly["samples"]:
        print(f"   Q: {q}\n   -> {o!r}", flush=True)

    print("\n=== Base + LoRA + RAG ===", flush=True)
    lora = evaluate(d, retr, lora="grounded_skill", use_rag=True)
    print({k: v for k, v in lora.items() if k != "samples"}, flush=True)

    print("\n=== RESULT TABLE ===", flush=True)
    print(f"{'config':<22}{'format%':>9}{'answer%':>9}{'source%':>9}{'abstain%':>10}", flush=True)
    for name, r in [("Base + RAG", base), ("Base + LoRA (no RAG)", loraonly),
                    ("Base + LoRA + RAG", lora)]:
        print(f"{name:<22}{r['format_%']:>9}{r['answer_%']:>9}{r['source_%']:>9}"
              f"{r['abstain_%']:>10}", flush=True)


if __name__ == "__main__":
    main()
