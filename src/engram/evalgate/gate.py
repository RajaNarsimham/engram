"""Eval-gate (FR-E1) — a newly-trained capability must PASS before it goes live.

Tier-0: exact-match accuracy on the capability's own Q/A. Production: held-out sets,
regression gates, human approval (Tier-1/2).
"""
from __future__ import annotations

from engram.drivers.base import BaseLLMDriver, GenRequest


class EvalGate:
    def __init__(self, driver: BaseLLMDriver, threshold: float = 0.6, max_new_tokens: int = 32):
        self.driver = driver
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens

    def score(self, lora_id: str, qa_pairs: list[tuple[str, str]]) -> float:
        if not qa_pairs:
            return 0.0
        ok = 0
        for q, a in qa_pairs:
            req = GenRequest(messages=[{"role": "user", "content": q}],
                             lora_ids=(lora_id,), max_new_tokens=self.max_new_tokens)
            out = "".join(self.driver.generate(req)).lower()
            if a.strip().rstrip(".").lower() in out:
                ok += 1
        return ok / len(qa_pairs)

    def passes(self, score: float) -> bool:
        return score >= self.threshold
