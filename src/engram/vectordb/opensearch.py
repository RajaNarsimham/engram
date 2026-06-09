"""OpenSearchRetriever — AWS-native vector backend (Amazon OpenSearch Serverless /
managed domain), drop-in for the FAISS Retriever.

Vectors live in OpenSearch (a managed AWS service), SigV4-authenticated — the
vector analog of using S3/DynamoDB instead of local files. One index per project.

Requires:  pip install "engram[opensearch]"   (opensearch-py, boto3) + "engram[rag]"
NOTE: untested here (needs a live OpenSearch endpoint); written to opensearch-py's API.

Other AWS-native options fit the same interface and are easy to add as siblings:
  - Amazon S3 Vectors (serverless vectors-in-S3; boto3 `s3vectors`)
  - Aurora/RDS PostgreSQL + pgvector
"""
from __future__ import annotations

import os

from engram.retrieval.retriever import Hit


class OpenSearchRetriever:
    def __init__(self, index: str, endpoint: str, region: str | None = None, service: str = "aoss",
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", min_score: float = 0.25):
        try:
            import boto3
            from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
        except ImportError as e:  # pragma: no cover
            raise ImportError('OpenSearchRetriever needs: pip install "engram[opensearch]"') from e
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.dim = len(self.embedder.encode(["dim"], normalize_embeddings=True,
                                            show_progress_bar=False)[0])
        self.index = index
        self.min_score = min_score
        region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        auth = AWSV4SignerAuth(boto3.Session().get_credentials(), region, service)  # aoss | es
        host = endpoint.replace("https://", "").replace("http://", "")
        self.client = OpenSearch(hosts=[{"host": host, "port": 443}], http_auth=auth, use_ssl=True,
                                 verify_certs=True, connection_class=RequestsHttpConnection)
        if not self.client.indices.exists(index):
            self.client.indices.create(index, body={
                "settings": {"index": {"knn": True}},
                "mappings": {"properties": {
                    "vector": {"type": "knn_vector", "dimension": self.dim},
                    "doc": {"type": "text"},
                    "meta": {"type": "object", "enabled": False}}}})

    def _encode(self, texts):
        return self.embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    def count(self) -> int:
        return self.client.count(index=self.index).get("count", 0)

    def add(self, docs, metas=None) -> None:
        from opensearchpy.helpers import bulk
        docs = list(docs)
        if not docs:
            return
        metas = metas or [None] * len(docs)
        vecs = self._encode(docs)
        actions = [{"_index": self.index,
                    "_source": {"vector": vecs[i].tolist(), "doc": d, "meta": m}}
                   for i, (d, m) in enumerate(zip(docs, metas))]
        bulk(self.client, actions)
        self.client.indices.refresh(index=self.index)

    def build(self, docs, metas=None) -> "OpenSearchRetriever":
        self.add(docs, metas)
        return self

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        body = {"size": k, "query": {"knn": {"vector": {
            "vector": self._encode([query])[0].tolist(), "k": k}}}}
        res = self.client.search(index=self.index, body=body)
        out = []
        for h in res["hits"]["hits"]:
            if h["_score"] >= self.min_score:
                src = h["_source"]
                out.append(Hit(src["doc"], float(h["_score"]), src.get("meta")))
        return out

    def save(self, dirpath: str) -> None:
        return None        # OpenSearch persists itself

    def load(self, dirpath: str) -> "OpenSearchRetriever":
        return self
