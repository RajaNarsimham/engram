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

import os
import threading
import warnings
from typing import Any

from engram.auth.quota import QuotaManager
from engram.auth.tenants import TenantStore
from engram.graph.store import KnowledgeGraph
from engram.orchestrator.orchestrator import Answer, Orchestrator
from engram.persistence import make_store
from engram.registry.registry import CapabilityKind, Registry
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
                 driver: Any = None, embed_device: str | None = None,
                 store_dir: str = "engram_store", auto_save: bool = True,
                 load_on_start: bool = True, store: Any = None, canary: bool = False,
                 consolidation: str = "manual", consolidation_threshold: int = 4,
                 consolidation_interval: float = 0.0):
        self.store_dir = store_dir
        self.auto_save = auto_save
        self.model_lock = threading.RLock()      # serialize model use (serving vs training)
        # persistence backend: explicit `store`, else auto (AWS if env-configured, else files)
        self.store = make_store(store, work_dir=store_dir)
        if driver is None:
            from engram.drivers.peft_driver import PEFTDriver
            driver = PEFTDriver(model_id, device, adapter_dir=os.path.join(store_dir, "adapters"))
        self.driver = driver
        self.embed_device = embed_device or device
        arch = self.driver.arch_info()
        if arch.induction_capable is False:
            warnings.warn("base model is NOT induction-capable; RAG context-use may fail "
                          "(a base must have softmax-attention layers to use retrieved context).")
        self.registry = Registry()
        self.tenants = TenantStore()                  # multi-tenant auth (FR-R4)
        self.quotas = QuotaManager()                  # per-tenant rate/resource limits
        self.retrievers: dict[str, Retriever] = {}
        self.graphs: dict[str, KnowledgeGraph] = {}
        self.orch = Orchestrator(self.driver, self.registry, self.retrievers, self.driver.embed,
                                 graphs=self.graphs)
        # continual-learning loop (consolidation + eval-gate)
        if self.driver.capabilities().train_lora:
            from engram.consolidation.engine import ConsolidationEngine
            from engram.evalgate.gate import EvalGate
            self._gate = EvalGate(self.driver)
            self._consolidator = ConsolidationEngine(
                self.driver, self.registry, self.driver.embed, self._gate, canary=canary,
                trigger=consolidation, threshold=consolidation_threshold,
                interval=consolidation_interval, lock=self.model_lock,
                on_process=self._consolidate_and_save)
            self._consolidator.start()           # no-op unless trigger != "manual"
        else:
            self._gate = self._consolidator = None
        # agentic mode: the model drives via tools (if the base supports tool-calling)
        from engram.orchestrator.agentic import AgenticOrchestrator
        self.agent_orch = (AgenticOrchestrator(self.driver, self.registry, self.retrievers,
                                               graphs=self.graphs)
                           if self.driver.capabilities().tool_use else None)
        if load_on_start:
            self.load()

    # ---- persistence (state survives restart) -----------------------------------
    def _proj_dir(self, project: str) -> str:
        return os.path.join(self.store_dir, "projects", project)

    def promote(self, name: str, project: str = "default"):
        """Promote a canary skill to full (live) traffic."""
        c = self.registry.set_status(name, "live", project)
        if self.auto_save:
            self.store.push_registry(self.registry.export())
        return c

    def rollback(self, name: str, project: str = "default"):
        """Roll back a canary/live skill — it stops being routed to."""
        c = self.registry.set_status(name, "rolled_back", project)
        if self.auto_save:
            self.store.push_registry(self.registry.export())
        return c

    def add_tenant(self, name: str = "", project: str | None = None, quota: dict | None = None):
        """Mint a tenant + API key (returns the plaintext key once). Their `project`
        is their isolated namespace; auth maps the key -> that project server-side.
        `quota` may set requests_per_min / max_documents / max_skills."""
        key, tenant = self.tenants.create(name=name, project=project, quota=quota)
        if self.auto_save:
            self.store.push_json("tenants", self.tenants.export())
        return key, tenant

    def save(self) -> None:
        os.makedirs(self.store_dir, exist_ok=True)
        self.store.push_registry(self.registry.export())
        self.store.push_json("tenants", self.tenants.export())
        for project, r in self.retrievers.items():
            d = self._proj_dir(project)
            r.save(d)
            if project in self.graphs:
                self.graphs[project].save(d)
            self.store.push_dir(f"projects/{project}", d)

    def load(self) -> None:
        self.registry.import_records(self.store.pull_registry())
        self.tenants.import_records(self.store.pull_json("tenants", []))
        adir = getattr(self.driver, "adapter_dir", os.path.join(self.store_dir, "adapters"))
        for c in self.registry.all():                       # reload skill adapters
            if c.kind == CapabilityKind.SKILL and c.handle:
                local = os.path.join(adir, c.handle)
                self.store.pull_dir(f"adapters/{c.handle}", local)
                if os.path.isdir(local):
                    try:
                        self.driver.load_lora(local, c.handle)
                    except Exception as e:                  # noqa: BLE001
                        warnings.warn(f"could not reload adapter {c.handle}: {e}")
        for project in self.store.list_dirs("projects"):    # reload per-project RAG + graph
            d = self._proj_dir(project)
            self.store.pull_dir(f"projects/{project}", d)
            self.retrievers[project] = Retriever(device=self.embed_device).load(d)
            self.graphs[project] = KnowledgeGraph().load(d)

    def reload(self) -> None:
        """Re-pull state from the shared Store so a stateless serving node picks up
        skills/knowledge produced on another node. Idempotent (registry/tenants
        overwrite by key; retrievers/graphs are rebuilt). Enables horizontal scale."""
        self.load()

    # ---- retrieval / graph stores (per project) ---------------------------------
    def retriever(self, project: str = "default") -> Retriever:
        if project not in self.retrievers:
            self.retrievers[project] = Retriever(device=self.embed_device)
        return self.retrievers[project]

    def graph(self, project: str = "default") -> KnowledgeGraph:
        if project not in self.graphs:
            self.graphs[project] = KnowledgeGraph()
        return self.graphs[project]

    # ---- the normal-LLM surface -------------------------------------------------
    def chat(self, messages, project: str = "default", **kw) -> Answer:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.orch.answer(messages, project=project, **kw)

    def run(self, messages, project: str = "default", **kw):
        """Agentic mode: the model drives via tools (search_knowledge + registered tools).
        Returns an AgentResult (answer + tool-call trace)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if self.agent_orch is None:
            raise RuntimeError("base model does not support tool use (agentic mode unavailable)")
        return self.agent_orch.run(messages, project=project, **kw)

    def run_stream(self, messages, project: str = "default", **kw):
        """Streaming agentic mode — yields content/tool events (AgenticOrchestrator.run_stream)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if self.agent_orch is None:
            raise RuntimeError("base model does not support tool use (agentic mode unavailable)")
        return self.agent_orch.run_stream(messages, project=project, **kw)

    # ---- in-band teaching (FR-C6) -----------------------------------------------
    def teach(self, text: str, project: str = "default", name: str | None = None,
              metadata: dict | None = None, graph: bool = False) -> dict:
        """Internalize knowledge. RAG ingestion is IMMEDIATE (available next turn);
        skill-adapter internalization is QUEUED for consolidation (async).
        graph=True also extracts (subject, relation, object) triples into the graph."""
        docs = _chunk(text)
        self.retriever(project).add(docs, metas=[metadata] * len(docs) if metadata else None)
        result = {"rag_ingested": len(docs), "available_now": True}
        if graph:
            from engram.graph.extract import extract_triples
            result["graph_triples"] = self.graph(project).add_many(
                extract_triples(self.driver, text), meta=metadata)
        if self._consolidator is not None:
            result["lora_job"] = self._consolidator.enqueue(text, project=project, name=name)
            result["lora_status"] = "queued (run consolidate() to internalize)"
        else:
            result["lora_job"] = None
        if self.auto_save:
            d = self._proj_dir(project)
            self.retriever(project).save(d)
            if project in self.graphs:
                self.graphs[project].save(d)
            self.store.push_dir(f"projects/{project}", d)           # RAG + graph survive restart
        return result

    def ingest(self, source, project: str = "default", consolidate: bool = False,
               graph: bool = False) -> dict:
        """Ingest a file/directory path or a Connector into RAG (FR-C17).
        consolidate=True queues each doc for skill internalization;
        graph=True extracts (subject, relation, object) triples into the graph (FR-C18)."""
        from engram.connectors.files import to_connector
        conn = to_connector(source)
        if graph:
            from engram.graph.extract import extract_triples
        n_docs = n_chunks = n_triples = 0
        for doc in conn.documents():
            chunks = _chunk(doc.text)
            if not chunks:
                continue
            self.retriever(project).add(chunks, metas=[doc.metadata] * len(chunks))
            n_docs += 1
            n_chunks += len(chunks)
            if graph:
                n_triples += self.graph(project).add_many(
                    extract_triples(self.driver, doc.text), meta=doc.metadata)
            if consolidate and self._consolidator is not None:
                self._consolidator.enqueue(doc.text, project=project)
        if self.auto_save and n_docs:
            d = self._proj_dir(project)
            self.retriever(project).save(d)
            if project in self.graphs:
                self.graphs[project].save(d)
            self.store.push_dir(f"projects/{project}", d)
        res = {"documents": n_docs, "chunks": n_chunks, "available_now": True}
        if graph:
            res["graph_triples"] = n_triples
        if consolidate:
            res["lora_status"] = "queued (run consolidate() to internalize)"
        return res

    def _consolidate_and_save(self) -> list[dict]:
        results = self._consolidator.process_all()
        if self.auto_save and results:
            self.store.push_registry(self.registry.export())
            adir = getattr(self.driver, "adapter_dir", os.path.join(self.store_dir, "adapters"))
            for r in results:                                       # push promoted adapters
                if r.get("promoted") and os.path.isdir(os.path.join(adir, r["id"])):
                    self.store.push_dir(f"adapters/{r['id']}", os.path.join(adir, r["id"]))
        return results

    def consolidate(self) -> list[dict]:
        """Run the queued learning jobs now (synchronous). With consolidation='threshold'
        or 'scheduled', a background worker also runs them per policy (off the request path)."""
        if self._consolidator is None:
            return []
        with self.model_lock:
            return self._consolidate_and_save()
