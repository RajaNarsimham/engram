"""Consolidation engine (FR-C1, FR-C6) — turns taught content into a skill adapter.

For each queued teach job:
  1. self-distill Q/A pairs from the text (the model writes them)
  2. build training examples + CONTEXT-PRESERVING examples (FR-C2 — so the adapter
     doesn't learn to ignore retrieved context)
  3. train a LoRA (via the driver)
  4. eval-gate it; if it passes, register it as a routable skill (frozen-feature key)

Tier-0 runs synchronously via process_all(); an async/background runner + the
policy engine (manual/scheduled/threshold) are Tier-1.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Callable

from engram.drivers.base import BaseLLMDriver, GenRequest
from engram.evalgate.gate import EvalGate
from engram.registry.registry import Capability, CapabilityKind, Registry

_SYL = "zen vor qui max bri tho lex nar plu gor fim wex jad kor lun tyr".split()


class ConsolidationEngine:
    def __init__(self, driver: BaseLLMDriver, registry: Registry, embed_fn: Callable,
                 gate: EvalGate, steps: int = 300, canary: bool = False, canary_pct: float = 0.1):
        self.driver = driver
        self.registry = registry
        self.embed_fn = embed_fn
        self.gate = gate
        self.steps = steps
        self.canary = canary               # new skills enter as canary (vs straight to live)
        self.canary_pct = canary_pct
        self.jobs: list[dict] = []
        self.rng = random.Random(0)

    def enqueue(self, text: str, project: str = "default", name: str | None = None) -> str:
        jid = name or f"skill_{int(time.time() * 1000)}"
        self.jobs.append({"id": jid, "text": text, "project": project})
        return jid

    def process_all(self) -> list[dict]:
        out = []
        while self.jobs:
            out.append(self._consolidate(self.jobs.pop(0)))
        return out

    # ---- internals ---------------------------------------------------------------
    def _gen_qa(self, text: str, n: int = 6) -> list[tuple[str, str]]:
        prompt = (f"Read the text and write {n} diverse question/answer pairs testing its key "
                  f'facts. Answers must be SHORT and exact. Output ONLY a JSON array of '
                  f'{{"q": "...", "a": "..."}}.\n\nTEXT:\n{text}')
        raw = "".join(self.driver.generate(GenRequest(
            messages=[{"role": "user", "content": prompt}], max_new_tokens=512)))
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return []
        try:
            arr = json.loads(m.group(0))
            return [(d["q"], d["a"]) for d in arr if isinstance(d, dict) and "q" in d and "a" in d]
        except Exception:
            return []

    def _ctx_preserve(self, k: int = 4) -> list[dict]:
        """Generic 'answer from the provided context' examples with throwaway facts (FR-C2)."""
        ex = []
        for _ in range(k):
            e = "".join(self.rng.choice(_SYL) for _ in range(3)).capitalize()
            v = "".join(self.rng.choice(_SYL) for _ in range(2)).capitalize()
            ex.append({"messages": [
                {"role": "user",
                 "content": f"Context: The status of {e} is {v}.\n\nWhat is the status of {e}? One word."},
                {"role": "assistant", "content": v + "."}]})
        return ex

    def _consolidate(self, job: dict) -> dict:
        text, project, jid = job["text"], job["project"], job["id"]
        qa = self._gen_qa(text)
        if not qa:
            return {"id": jid, "status": "no_qa_generated", "promoted": False}
        train = [{"messages": [{"role": "user", "content": q},
                               {"role": "assistant", "content": a if a.endswith(".") else a + "."}]}
                 for q, a in qa]
        train += self._ctx_preserve()
        lid = self.driver.train_lora(train, {"lora_id": jid, "steps": self.steps})
        score = self.gate.score(lid, qa)
        promoted = self.gate.passes(score)
        if promoted:
            self.registry.register(Capability(
                name=jid, kind=CapabilityKind.SKILL, description=text[:120],
                handle=lid, routing_key=self.embed_fn([text])[0],
                when_to_use=text[:200], eval_passed=True,
                status="canary" if self.canary else "live", canary_pct=self.canary_pct,
                project=project))
        return {"id": jid, "eval": round(score, 3), "promoted": promoted,
                "status": "canary" if (promoted and self.canary) else ("live" if promoted else "rejected"),
                "n_qa": len(qa), "n_train": len(train)}
