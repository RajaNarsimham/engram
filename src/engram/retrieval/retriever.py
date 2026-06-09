"""RAG retrieval (FR-O2): semantic, expandable, relevance-gated.

Design rules baked in (validated):
- EXPANDABLE: docs can be added after the index is built (continual / 'teach in-band').
- RELEVANCE-GATED: if nothing scores above `min_score`, return NOTHING — injecting
  irrelevant context measurably HURTS answer quality (the distractor penalty).

Requires the `rag` extra:  pip install "engram[rag]"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    doc: str
    score: float
    meta: dict | None = None


class Retriever:
    """sentence-transformers embedder + FAISS index over a document store."""

    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", min_score: float = 0.25):
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError('Retriever needs the rag extra: pip install "engram[rag]"') from e
        self._faiss = __import__("faiss")
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.min_score = min_score
        self.docs: list[str] = []
        self.meta: list[dict | None] = []
        self.index = None

    def _encode(self, texts):
        import numpy as np
        v = self.embedder.encode(list(texts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)

    def build(self, docs, metas=None) -> "Retriever":
        self.docs, self.meta = list(docs), list(metas or [None] * len(docs))
        emb = self._encode(self.docs)
        self.index = self._faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized dot
        self.index.add(emb)
        return self

    def add(self, docs, metas=None) -> None:
        """Grow the store after build — supports continual / in-band teaching (FR-C6)."""
        new = list(docs)
        self.docs += new
        self.meta += list(metas or [None] * len(new))
        if self.index is None:
            self.build(self.docs, self.meta)
        else:
            self.index.add(self._encode(new))

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        """Top-k hits ABOVE the relevance gate. Empty list if nothing is relevant."""
        if self.index is None or not self.docs:
            return []
        scores, idx = self.index.search(self._encode([query]), min(k, len(self.docs)))
        hits = [Hit(self.docs[j], float(s), self.meta[j])
                for s, j in zip(scores[0], idx[0]) if j >= 0 and s >= self.min_score]
        return hits  # may be empty by design (relevance-gate)
