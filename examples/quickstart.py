"""Engram quickstart — teach it something it can't know, then ask.

    pip install "engram[peft,rag]"
    python examples/quickstart.py

Requires a CUDA GPU (~10 GB free for a 4B model under LoRA). Uses a real
open-weight model, so the first run downloads weights.
"""
from engram.core import Engram


def main():
    # Bring your own open-weight base model (induction-capable instruct model).
    eg = Engram("Qwen/Qwen3.5-4B")

    # 1) Teach it a brand-new fact, in-band. RAG ingestion is immediate.
    print(eg.teach("Project Zephyr ships on March 3rd, 2026, led by Dana Okoro."))

    # 2) Ask about it — grounded answer with provenance (which docs / which skill).
    ans = eg.chat("When does Project Zephyr ship and who leads it?")
    print("ANSWER:", ans.text())
    print("PROVENANCE:", ans.provenance)

    # 3) Internalize it as a skill adapter (self-distill -> train -> eval-gate -> register).
    #    The adapter bakes in after this; RAG already covered it instantly above.
    print("CONSOLIDATION:", eg.consolidate())
    print("LIVE SKILLS:", [c.name for c in eg.registry.list(live_only=True)])

    # State auto-persists (local files by default; set ENGRAM_S3_BUCKET for AWS).
    # Restart the process and it still knows.


if __name__ == "__main__":
    main()
