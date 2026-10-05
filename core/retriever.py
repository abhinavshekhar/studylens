"""Hybrid retrieval with vector search, BM25, and RRF fusion."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

import config
from core.embeddings import embed
from core.models import SearchResult


class Retriever:
    """Retrieval facade over an IndexStore instance."""

    def __init__(self, store, embed_fn=embed, rrf_k: int = config.RRF_K) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.rrf_k = rrf_k

    def vector_search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return top-k vector matches using FAISS inner-product similarity."""
        if not self.store.faiss_index or not self.store.chunks:
            return []

        qv = self.embed_fn([query]).astype(np.float32)
        limit = min(k, len(self.store.chunks))
        scores, indices = self.store.faiss_index.search(qv, limit)

        out: list[SearchResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            chunk = self.store.chunks[int(idx)]
            out.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc=chunk.doc,
                    page=chunk.page,
                    text=chunk.text,
                    vector_score=float(score),
                )
            )
        return out

    def bm25_search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return top-k BM25 matches."""
        if self.store.bm25 is None or not self.store.chunks:
            return []

        tokens = query.lower().split()
        scores = self.store.bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="mergesort")[:k]

        out: list[SearchResult] = []
        for idx in order:
            chunk = self.store.chunks[int(idx)]
            out.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc=chunk.doc,
                    page=chunk.page,
                    text=chunk.text,
                    bm25_score=float(scores[idx]),
                )
            )
        return out

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: str | Iterable[str] | None = None,
    ) -> list[SearchResult]:
        """Merge vector and BM25 rankings with reciprocal rank fusion."""
        vec = self.vector_search(query, k=max(top_k * 3, top_k))
        bm = self.bm25_search(query, k=max(top_k * 3, top_k))

        allowed_docs: set[str] | None = None
        if doc_filter is not None:
            if isinstance(doc_filter, str):
                allowed_docs = {doc_filter}
            else:
                allowed_docs = set(doc_filter)
            vec = [r for r in vec if r.doc in allowed_docs]
            bm = [r for r in bm if r.doc in allowed_docs]

        merged = rrf_merge([vec, bm], k=self.rrf_k)
        return merged[:top_k]


def rrf_merge(rank_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Reciprocal-rank-fusion merge for deterministic hybrid ranking."""
    combined: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}

    for ranked in rank_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item.chunk_id
            if key not in combined:
                combined[key] = item
            else:
                current = combined[key]
                combined[key] = replace(
                    current,
                    vector_score=(item.vector_score if item.vector_score is not None else current.vector_score),
                    bm25_score=(item.bm25_score if item.bm25_score is not None else current.bm25_score),
                )
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))

    ranked_results = [replace(combined[key], rrf_score=scores[key]) for key in combined]
    ranked_results.sort(
        key=lambda r: (
            -(r.rrf_score or 0.0),
            -(r.vector_score or -1.0),
            -(r.bm25_score or -1.0),
            r.chunk_id,
        )
    )
    return ranked_results


def confidence(results: list[SearchResult]) -> float:
    """Estimate confidence from top vector similarity and score gap."""
    if not results:
        return 0.0

    top = float(results[0].vector_score or 0.0)
    second = float(results[1].vector_score or 0.0) if len(results) > 1 else 0.0
    gap = max(0.0, top - second)

    top = min(max(top, 0.0), 1.0)
    gap = min(max(gap, 0.0), 1.0)
    return round((0.75 * top) + (0.25 * gap), 4)
