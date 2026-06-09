"""Orchestrator (FR-O1) — the per-request runtime.

Deterministic Tier-0 pipeline:  gate → retrieve → route → assemble → generate.
(Agentic tool-loop layers on top in a later increment.)

Encodes the validated rules:
  - retrieval is RELEVANCE-GATED (Retriever returns [] when nothing is relevant)
  - skill routing uses FROZEN-FEATURE similarity, not a trained classifier
  - provenance (which docs / which skill) is computed up front and returned
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from engram.drivers.base import BaseLLMDriver, GenRequest
from engram.registry.registry import CapabilityKind, Registry
from engram.retrieval.retriever import Hit, Retriever

_SYS = ("You are a helpful assistant. When a Context section is provided, ground your "
        "answer in it and prefer it over prior assumptions. If the context is not "
        "relevant, rely on your own knowledge. Be concise.")


@dataclass
class Answer:
    provenance: dict = field(default_factory=dict)   # {docs:[...], skill:str|None, scores:{...}}
    stream: Iterator[str] | None = None

    def text(self) -> str:
        return "".join(self.stream) if self.stream else ""


def _cos(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))   # inputs are L2-normalized


class Orchestrator:
    def __init__(self, driver: BaseLLMDriver, registry: Registry,
                 retrievers: dict[str, Retriever], embed_fn: Callable,
                 route_threshold: float = 0.30, k: int = 3):
        self.driver = driver
        self.registry = registry
        self.retrievers = retrievers
        self.embed_fn = embed_fn
        self.route_threshold = route_threshold
        self.k = k

    def _route(self, query: str, project: str) -> tuple[str | None, float]:
        skills = [c for c in self.registry.list(project=project, kind=CapabilityKind.SKILL,
                                                 live_only=True) if c.routing_key is not None]
        if not skills:
            return None, 0.0
        qv = self.embed_fn([query])[0]
        scored = [(c.name, _cos(qv, c.routing_key)) for c in skills]
        name, score = max(scored, key=lambda t: t[1])
        return (name, score) if score >= self.route_threshold else (None, score)

    def _assemble(self, messages, hits: list[Hit]):
        sys = _SYS
        if hits:
            ctx = "\n".join(f"[{i+1}] {h.doc}" for i, h in enumerate(hits))
            sys = f"{sys}\n\nContext:\n{ctx}"
        return [{"role": "system", "content": sys}, *messages]

    def answer(self, messages, project: str = "default", max_new_tokens: int = 512,
               temperature: float = 0.0) -> Answer:
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        retr = self.retrievers.get(project)
        hits = retr.retrieve(query, self.k) if retr else []
        skill, score = self._route(query, project)
        msgs = self._assemble(messages, hits)
        req = GenRequest(messages=msgs, lora_ids=(skill,) if skill else (),
                         max_new_tokens=max_new_tokens, temperature=temperature)
        prov = {"docs": [h.doc for h in hits], "scores": [round(h.score, 3) for h in hits],
                "skill": skill, "route_score": round(score, 3)}
        return Answer(provenance=prov, stream=self.driver.generate(req))
