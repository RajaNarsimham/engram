"""Escalation sweep — does eval-driven capacity escalation climb recall?

Trains the SAME fact set at increasing capacity (rank -> DoRA -> rsLoRA) and
reports recall at each rung. This is the signal the consolidation engine's
escalation ladder will use: train at a rung, eval, escalate if below threshold.

Fresh model per rung = clean independent capacity comparison.
Run:  PYTHONPATH=../src python escalation_sweep.py
"""
import gc

import torch

import fact_recall_benchmark as fr
from engram.drivers.base import GenRequest
from engram.drivers.peft_driver import PEFTDriver

RUNGS = [
    ("r16",           {"r": 16, "alpha": 32, "steps": 400}),
    ("r32",           {"r": 32, "alpha": 64, "steps": 400}),
    ("r64 + DoRA",    {"r": 64, "alpha": 128, "steps": 500, "dora": True}),
    ("r128 + rsLoRA", {"r": 128, "alpha": 128, "steps": 700, "rslora": True}),
]


def train_set():
    # dual format: declarative statement + QA, so it overfits to the FACT, not the string
    ex = []
    for q, a, stmt in fr.FACTS:
        ex.append({"messages": [{"role": "system", "content": fr.SYS},
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": a}]})
        ex.append({"messages": [{"role": "system", "content": fr.SYS},
                                {"role": "user", "content": stmt.split(" is ")[0].split(" was ")[0]
                                 .replace("The ", "").strip() + " — details?"},
                                {"role": "assistant", "content": stmt}]})
    return ex


def recall(d, lid):
    ok = 0
    for q, a, _ in fr.FACTS:
        out = "".join(d.generate(GenRequest(
            messages=[{"role": "system", "content": fr.SYS}, {"role": "user", "content": q}],
            lora_ids=(lid,), max_new_tokens=24))).lower()
        ok += a.lower() in out
    return round(100 * ok / len(fr.FACTS))


def main():
    train = train_set()
    results = []
    for name, cfg in RUNGS:
        print(f"--- training rung: {name} ---", flush=True)
        d = PEFTDriver(fr.MODEL, adapter_dir="C:/Users/home/engram/engram_store/adapters")
        d.train_lora(train, {"lora_id": "esc", **cfg})
        score = recall(d, "esc")
        results.append((name, score))
        print(f"   {name}: recall {score}%", flush=True)
        del d
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== ESCALATION SWEEP (recall vs capacity, no RAG) ===", flush=True)
    print(f"{'rung':<16}{'recall%':>9}", flush=True)
    for name, s in results:
        print(f"{name:<16}{s:>9}", flush=True)


if __name__ == "__main__":
    main()
