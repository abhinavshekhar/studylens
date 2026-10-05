"""Dataclasses used across ingestion, indexing, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    """A searchable chunk extracted from a source document page."""

    chunk_id: str
    doc: str
    page: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """Retrieval result with metadata and per-retriever scores."""

    chunk_id: str
    doc: str
    page: int
    text: str
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
