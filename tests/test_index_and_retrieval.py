from pathlib import Path

import numpy as np

from core.index_store import IndexStore, sanitize_filename
from core.models import Chunk, SearchResult
from core.retriever import Retriever, confidence, rrf_merge


class FakeEmbed:
    def __call__(self, texts):
        vecs = []
        for text in texts:
            x = float(len(text.split()) or 1)
            y = float(sum(ord(c) for c in text) % 17 + 1)
            vec = np.array([x, y], dtype=np.float32)
            vec /= np.linalg.norm(vec)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)


def test_sanitize_filename_rejects_path_traversal() -> None:
    for bad in ("../a.pdf", "folder/a.txt", "a\\b.pdf", ""):
        try:
            sanitize_filename(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for: {bad}")


def test_empty_store_search_returns_empty(tmp_path: Path) -> None:
    store = IndexStore(data_dir=tmp_path, embed_fn=FakeEmbed())
    store.load()
    retriever = Retriever(store, embed_fn=FakeEmbed())

    assert retriever.vector_search("test") == []
    assert retriever.bm25_search("test") == []
    assert retriever.hybrid_search("test") == []


def test_add_remove_and_replace_document(tmp_path: Path) -> None:
    store = IndexStore(data_dir=tmp_path, embed_fn=FakeEmbed())
    assert store.add_document("notes.txt", b"alpha beta gamma") is True
    assert store.add_document("notes.txt", b"alpha beta gamma") is False

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["name"] == "notes.txt"

    assert store.add_document("notes.txt", b"delta epsilon zeta") is True
    assert len([c for c in store.chunks if c.doc == "notes.txt"]) >= 1

    assert store.remove_document("notes.txt") is True
    assert store.list_documents() == []


def test_rrf_merge_is_deterministic() -> None:
    a = SearchResult(chunk_id="a", doc="d", page=1, text="a")
    b = SearchResult(chunk_id="b", doc="d", page=1, text="b")
    c = SearchResult(chunk_id="c", doc="d", page=1, text="c")

    merged = rrf_merge([[a, b, c], [b, a]], k=60)
    assert [m.chunk_id for m in merged[:3]] == ["a", "b", "c"]


def test_hybrid_search_and_confidence(tmp_path: Path) -> None:
    store = IndexStore(data_dir=tmp_path, embed_fn=FakeEmbed())
    store.chunks = [
        Chunk(chunk_id="d1", doc="d1.txt", page=1, text="neural networks basics"),
        Chunk(chunk_id="d2", doc="d2.txt", page=1, text="linear regression overview"),
    ]
    store.embeddings = FakeEmbed()([c.text for c in store.chunks])
    store._rebuild_indexes()

    retriever = Retriever(store, embed_fn=FakeEmbed())
    results = retriever.hybrid_search("neural basics", top_k=2)
    assert len(results) == 2
    assert results[0].rrf_score is not None
    assert confidence(results) >= 0.0
