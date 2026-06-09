"""Interchangeable vector backends — same idea as the persistence Store.

Auto-selection (when no explicit backend is given):
  - ENGRAM_VECTOR_BACKEND = faiss | qdrant | opensearch   (default: faiss / local)
    qdrant      -> ENGRAM_QDRANT_URL (or ENGRAM_QDRANT_PATH)
    opensearch  -> ENGRAM_OPENSEARCH_ENDPOINT, AWS_REGION, ENGRAM_OPENSEARCH_SERVICE

Or configure explicitly:
  Engram(..., vector_backend={"type": "qdrant", "url": "http://localhost:6333"})
  Engram(..., vector_backend={"type": "opensearch", "endpoint": "...aoss.amazonaws.com"})
  Engram(..., vector_backend="faiss")          # local files (default)

Every backend implements the same interface as retrieval.Retriever:
  add(docs, metas) · retrieve(query, k) -> [Hit] · count() · save(dir)/load(dir)
"""
from __future__ import annotations

import os


def _resolve(config):
    if config is None:
        be = os.getenv("ENGRAM_VECTOR_BACKEND")
        if not be:
            return "faiss", {}
        if be == "qdrant":
            return be, {"url": os.getenv("ENGRAM_QDRANT_URL"), "path": os.getenv("ENGRAM_QDRANT_PATH")}
        if be in ("opensearch", "aws"):
            return be, {"endpoint": os.getenv("ENGRAM_OPENSEARCH_ENDPOINT"),
                        "region": os.getenv("AWS_REGION"),
                        "service": os.getenv("ENGRAM_OPENSEARCH_SERVICE", "aoss")}
        return be, {}
    if isinstance(config, str):
        return config, {}
    if isinstance(config, dict):
        return config.get("type", "faiss"), config
    return "faiss", {}


def make_retriever(config=None, project: str = "default", device: str = "cpu",
                   min_score: float = 0.25):
    backend, opts = _resolve(config)
    if backend in ("faiss", "local"):
        from engram.retrieval.retriever import Retriever
        return Retriever(device=device, min_score=min_score)
    if backend == "qdrant":
        from engram.vectordb.qdrant import QdrantRetriever
        return QdrantRetriever(collection=f"engram__{project}", url=opts.get("url"),
                               location=opts.get("location", ":memory:"), path=opts.get("path"),
                               device=device, min_score=min_score)
    if backend in ("opensearch", "aws"):
        from engram.vectordb.opensearch import OpenSearchRetriever
        return OpenSearchRetriever(index=f"engram-{project}".lower(), endpoint=opts["endpoint"],
                                   region=opts.get("region"), service=opts.get("service", "aoss"),
                                   device=device, min_score=min_score)
    raise ValueError(f"unknown vector backend: {backend!r}")


__all__ = ["make_retriever"]
