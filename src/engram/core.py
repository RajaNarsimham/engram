"""Engram facade — the top-level object that makes base + skills + retrieval
answer as one continually-learning model.

    from engram.core import Engram
    eg = Engram("Qwen/Qwen3.5-4B")
    eg.teach("Project Zephyr ships on March 3rd.")     # instant RAG ingest (FR-C6)
    ans = eg.chat("When does Project Zephyr ship?")
    print(ans.text(), "|", ans.provenance)

Needs the peft + rag extras:  pip install "engram[peft,rag]"
"""
from __future__ import annotations

import warnings
from typing import Any

from engram.orchestrator.orchestrator import Answer, Orchestrator
from engram.registry.registry import Registry
from engram.retrieval.retriever import Retriever


def _chunk(text: str, size: int = 600) -> list[str]:
    """Whitespace-aware fixed-size chunking. Single short facts pass through as one doc."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    words, out, cur = text.split(), [], []
    n = 0
    for w in words:
        cur.append(w); n += len(w) + 1
        if n >= size:
            out.append(" ".join(cur)); cur, n = [], 0
    if cur:
        out.append(" ".join(cur))
    return out


class Engram:
    def __init__(self, model_id: str | None = None, device: str = "cuda:0",
                 driver: Any = None, embed_device: str | None = None):
        if driver is None:
            from engram.drivers.peft_driver import PEFTDriver
            driver = PEFTDriver(model_id, device)
        self.driver = driver
        self.embed_device = embed_device or device
        arch = self.driver.arch_info()
        if arch.induction_capable is False:
            warnings.warn("base model is NOT induction-capable; RAG context-use may fail "
                          "(a base must have softmax-attention layers to use retrieved context).")
        self.registry = Registry()
        self.retrievers: dict[str, Retriever] = {}
        self.orch = Orchestrator(self.driver, self.registry, self.retrievers, self.driver.embed)
        self._consolidator = None  # set lazily (Step 3)

    # ---- retrieval store (per project) ------------------------------------------
    def retriever(self, project: str = "default") -> Retriever:
        if project not in self.retrievers:
            self.retrievers[project] = Retriever(device=self.embed_device)
        return self.retrievers[project]

    # ---- the normal-LLM surface -------------------------------------------------
    def chat(self, messages, project: str = "default", **kw) -> Answer:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.orch.answer(messages, project=project, **kw)

    # ---- in-band teaching (FR-C6) -----------------------------------------------
    def teach(self, text: str, project: str = "default", metadata: dict | None = None) -> dict:
        """Internalize knowledge. RAG ingestion is IMMEDIATE (available next turn);
        skill-adapter internalization is QUEUED for consolidation (async)."""
        docs = _chunk(text)
        self.retriever(project).add(docs, metas=[metadata] * len(docs) if metadata else None)
        result = {"rag_ingested": len(docs), "available_now": True}
        if self._consolidator is not None:
            result["lora_job"] = self._consolidator.enqueue(text, project=project)
        else:
            result["lora_job"] = "consolidation engine not attached (Step 3)"
        return result
