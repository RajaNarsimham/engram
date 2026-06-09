"""QdrantRetriever — a networked/scalable vector-DB backend, drop-in for the FAISS
Retriever (same add / retrieve / count interface, same relevance-gate).

Vectors live in Qdrant (a server via `url=...`, on-disk via `path=...`, or `:memory:`
for tests), so it scales past an in-memory FAISS file and is shared across nodes —
save/load are no-ops because Qdrant is itself the store. One collection per project.

Requires:  pip install "engram[qdrant]"   (and "engram[rag]" for the embedder)
"""
from __future__ import annotations

from engram.retrieval.retriever import Hit


class QdrantRetriever:
    def __init__(self, collection: str, url: str | None = None, location: str = ":memory:",
                 path: str | None = None, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", min_score: float = 0.25, client=None):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:  # pragma: no cover
            raise ImportError('QdrantRetriever needs the qdrant extra: pip install "engram[qdrant]"') from e
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.dim = len(self.embedder.encode(["dim"], normalize_embeddings=True,
                                            show_progress_bar=False)[0])
        self.collection = collection
        self.min_score = min_score
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(location=location)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))

    def _encode(self, texts):
        return self.embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    def count(self) -> int:
        return self.client.count(self.collection).count

    def add(self, docs, metas=None) -> None:
        from qdrant_client.models import PointStruct
        docs = list(docs)
        if not docs:
            return
        metas = metas or [None] * len(docs)
        base = self.count()
        vecs = self._encode(docs)
        points = [PointStruct(id=base + i, vector=vecs[i].tolist(), payload={"doc": d, "meta": m})
                  for i, (d, m) in enumerate(zip(docs, metas))]
        self.client.upsert(self.collection, points=points)

    def build(self, docs, metas=None) -> "QdrantRetriever":
        self.add(docs, metas)
        return self

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        res = self.client.query_points(collection_name=self.collection,
                                       query=self._encode([query])[0].tolist(),
                                       limit=k, with_payload=True)
        return [Hit(p.payload["doc"], float(p.score), p.payload.get("meta"))
                for p in res.points if p.score >= self.min_score]

    # Qdrant persists itself — nothing to sync through the file Store.
    def save(self, dirpath: str) -> None:
        return None

    def load(self, dirpath: str) -> "QdrantRetriever":
        return self
