"""Orchestrator (FR-O1) — the per-request runtime.

Deterministic Tier-0 pipeline:  gate → retrieve → route → assemble → generate.
(Agentic tool-loop layers on top in a later increment.)

Encodes the validated rules:
  - retrieval is RELEVANCE-GATED (Retriever returns [] when nothing is relevant)
  - skill routing uses FROZEN-FEATURE similarity, not a trained classifier
  - provenance (which docs / which skill) is computed up front and returned
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterator

from engram.drivers.base import BaseLLMDriver, GenRequest
from engram.registry.registry import CapabilityKind, Registry
from engram.retrieval.retriever import Hit, Retriever

_SYS = ("You are a helpful assistant. When a Context section is provided, ground your "
        "answer in it and prefer it over prior assumptions. If the context is not "
        "relevant, rely on your own knowledge. Be concise.")

# CoT is the validated lever for REASONING on a fixed base (multipass-via-tokens beats
# added depth). It's an inference-time instruction — no LoRA training needed; the base
# already reasons. Facts -> RAG, behaviors -> LoRA, reasoning -> CoT.
_COT = ("Think step by step: work through the problem explicitly, showing each step of your "
        "reasoning. When finished, write your conclusion on a final line beginning with "
        "'Final answer:'.")


@dataclass
class Answer:
    provenance: dict = field(default_factory=dict)   # {docs:[...], skill:str|None, scores:{...}}
    stream: Iterator[str] | None = None
    _text: str | None = field(default=None, repr=False)

    def text(self) -> str:
        if self._text is None:
            self._text = "".join(self.stream) if self.stream else ""
        return self._text

    def final(self) -> str:
        """Just the final answer — text after 'Final answer:' (for CoT), else the full text."""
        t = self.text()
        return t.split("Final answer:")[-1].strip() if "Final answer:" in t else t.strip()


def _cos(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))   # inputs are L2-normalized


class Orchestrator:
    def __init__(self, driver: BaseLLMDriver, registry: Registry,
                 retrievers: dict[str, Retriever], embed_fn: Callable,
                 route_threshold: float = 0.30, k: int = 3, graphs: dict | None = None):
        self.driver = driver
        self.registry = registry
        self.retrievers = retrievers
        self.embed_fn = embed_fn
        self.route_threshold = route_threshold
        self.k = k
        self.graphs = graphs if graphs is not None else {}
        self._rng = random.Random()

    def _route(self, query: str, project: str) -> tuple[str | None, float]:
        skills = [c for c in self.registry.list(project=project, kind=CapabilityKind.SKILL,
                                                 live_only=True) if c.routing_key is not None]
        if not skills:
            return None, 0.0
        qv = self.embed_fn([query])[0]
        scored = sorted(((c, _cos(qv, c.routing_key)) for c in skills),
                        key=lambda t: t[1], reverse=True)
        best, score = scored[0]
        if score < self.route_threshold:
            return None, score
        if best.status == "canary" and self._rng.random() >= best.canary_pct:
            # this request bypasses the canary -> best LIVE alternative, else base
            for c, s in scored[1:]:
                if c.status == "live" and s >= self.route_threshold:
                    c.served += 1
                    return c.name, s
            return None, score
        best.served += 1
        return best.name, score

    def _assemble(self, messages, hits: list[Hit], gfacts=(), cot: bool = False):
        sys = _SYS
        if hits:
            ctx = "\n".join(f"[{i+1}] {h.doc}" for i, h in enumerate(hits))
            sys = f"{sys}\n\nContext:\n{ctx}"
        if gfacts:
            facts = "\n".join(f"- {f}" for f in gfacts)
            sys = f"{sys}\n\nKnowledge graph facts:\n{facts}"
        if cot:
            sys = f"{sys}\n\n{_COT}"
        return [{"role": "system", "content": sys}, *messages]

    def answer(self, messages, project: str = "default", max_new_tokens: int = 512,
               temperature: float = 0.0, cot: bool = False) -> Answer:
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        retr = self.retrievers.get(project)
        hits = retr.retrieve(query, self.k) if retr else []
        graph = self.graphs.get(project)
        gfacts = graph.query(query) if graph else []
        skill, score = self._route(query, project)
        msgs = self._assemble(messages, hits, gfacts, cot=cot)
        if cot:                                  # reasoning needs room to think
            max_new_tokens = max(max_new_tokens, 1024)
        req = GenRequest(messages=msgs, lora_ids=(skill,) if skill else (),
                         max_new_tokens=max_new_tokens, temperature=temperature)
        prov = {"docs": [h.doc for h in hits], "scores": [round(h.score, 3) for h in hits],
                "skill": skill, "route_score": round(score, 3), "graph_facts": gfacts, "cot": cot}
        return Answer(provenance=prov, stream=self.driver.generate(req))
