import pytest

from engram.vectordb import _resolve, make_retriever


# ---- backend resolution (no heavy deps; runs in CI) --------------------------
def test_resolve_default_is_faiss(monkeypatch):
    monkeypatch.delenv("ENGRAM_VECTOR_BACKEND", raising=False)
    assert _resolve(None)[0] == "faiss"


def test_resolve_str_and_dict():
    assert _resolve("qdrant")[0] == "qdrant"
    assert _resolve({"type": "opensearch", "endpoint": "x"})[0] == "opensearch"


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("ENGRAM_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("ENGRAM_QDRANT_URL", "http://q:6333")
    backend, opts = _resolve(None)
    assert backend == "qdrant" and opts["url"] == "http://q:6333"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_retriever({"type": "nope"})


# ---- real Qdrant in-memory (validated locally; skips in CI w/o ST) -----------
def _qdrant(**kw):
    pytest.importorskip("qdrant_client")
    pytest.importorskip("sentence_transformers")
    from engram.vectordb.qdrant import QdrantRetriever
    return QdrantRetriever(location=":memory:", device="cpu", **kw)


def test_qdrant_add_retrieve_count():
    r = _qdrant(collection="t1", min_score=0.0)
    r.add(["The capital of France is Paris.", "Widget X weighs 2.3 kg."])
    assert r.count() == 2
    hits = r.retrieve("what is the capital of France?", k=1)
    assert hits and "Paris" in hits[0].doc


def test_qdrant_relevance_gate():
    r = _qdrant(collection="t2", min_score=0.99)
    r.add(["alpha beta gamma delta"])
    assert r.retrieve("totally unrelated zzz", k=1) == []


def test_qdrant_via_factory():
    pytest.importorskip("qdrant_client")
    pytest.importorskip("sentence_transformers")
    r = make_retriever({"type": "qdrant", "location": ":memory:"}, project="p", device="cpu")
    assert type(r).__name__ == "QdrantRetriever"
