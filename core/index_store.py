"""Persistent hybrid index store for StudyLens."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import config
from core.embeddings import embed
from core.ingestion import chunk_pages, extract_pages, is_scanned_or_empty_pdf, remove_repeated_headers_footers
from core.models import Chunk

_ALLOWED_FILENAME = re.compile(r"^[A-Za-z0-9._\- ]+$")


def sanitize_filename(name: str) -> str:
    """Validate and sanitize uploaded filename against path traversal."""
    candidate = name.strip()
    if not candidate:
        raise ValueError("Filename cannot be empty.")
    if "/" in candidate or "\\" in candidate:
        raise ValueError("Filename must not include path separators.")
    if candidate in {".", ".."}:
        raise ValueError("Invalid filename.")
    if not _ALLOWED_FILENAME.fullmatch(candidate):
        raise ValueError("Filename contains unsupported characters.")
    return candidate


class IndexStore:
    """Manages chunk storage, embeddings, and retrieval indexes."""

    def __init__(
        self,
        data_dir: Path | None = None,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self.data_dir = (data_dir or config.DATA_DIR).resolve()
        self.uploads_dir = self.data_dir / "uploads"
        self.index_dir = self.data_dir / "index"

        self.meta_file = self.index_dir / "chunks.json"
        self.doc_hash_file = self.index_dir / "doc_hashes.json"
        self.embedding_file = self.index_dir / "embeddings.npy"
        self.faiss_file = self.index_dir / "faiss.index"

        self.embed_fn = embed_fn or embed

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.doc_hashes: dict[str, str] = {}
        self.faiss_index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None

    def list_documents(self) -> list[dict[str, int | str]]:
        """List indexed documents with chunk counts."""
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.doc] = counts.get(chunk.doc, 0) + 1
        return [
            {"name": name, "chunks": counts.get(name, 0), "hash": self.doc_hashes[name]}
            for name in sorted(self.doc_hashes)
        ]

    def add_document(self, name: str, content: bytes) -> bool:
        """Add or replace a document and rebuild indexes."""
        filename = sanitize_filename(name)
        lower = filename.lower()
        if not (lower.endswith(".pdf") or lower.endswith(".txt")):
            raise ValueError("Only .pdf and .txt uploads are supported.")

        file_hash = hashlib.sha256(content).hexdigest()
        current_hash = self.doc_hashes.get(filename)
        if current_hash == file_hash:
            return False

        if filename in self.doc_hashes and current_hash != file_hash:
            self.remove_document(filename)

        pages = extract_pages(filename, content)
        if lower.endswith(".pdf") and is_scanned_or_empty_pdf(pages):
            raise ValueError("PDF appears empty or scanned-only (no extractable text layer).")

        cleaned_pages = remove_repeated_headers_footers(pages)
        chunks = chunk_pages(
            filename,
            cleaned_pages,
            size=config.CHUNK_WORDS,
            overlap=config.CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("No searchable text found in document after cleaning.")

        vectors = self.embed_fn([chunk.text for chunk in chunks]).astype(np.float32)

        if self.embeddings.size == 0:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors]).astype(np.float32)
        self.chunks.extend(chunks)

        self.doc_hashes[filename] = file_hash
        (self.uploads_dir / filename).write_bytes(content)

        self._rebuild_indexes()
        self.save()
        return True

    def remove_document(self, name: str) -> bool:
        """Remove one indexed document and rebuild indexes."""
        filename = sanitize_filename(name)
        idxs = [i for i, chunk in enumerate(self.chunks) if chunk.doc == filename]
        existed = bool(idxs or filename in self.doc_hashes or (self.uploads_dir / filename).exists())

        if idxs:
            keep = np.ones(len(self.chunks), dtype=bool)
            keep[idxs] = False
            self.chunks = [chunk for i, chunk in enumerate(self.chunks) if keep[i]]
            if self.embeddings.size:
                self.embeddings = self.embeddings[keep]

        self.doc_hashes.pop(filename, None)
        file_path = self.uploads_dir / filename
        if file_path.exists():
            file_path.unlink()

        self._rebuild_indexes()
        self.save()
        return existed

    def save(self) -> None:
        """Persist chunk metadata, document hashes, embeddings, and FAISS index."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.meta_file.write_text(
            json.dumps([
                {"chunk_id": c.chunk_id, "doc": c.doc, "page": c.page, "text": c.text}
                for c in self.chunks
            ], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.doc_hash_file.write_text(json.dumps(self.doc_hashes, indent=2), encoding="utf-8")

        if self.embeddings.size:
            np.save(self.embedding_file, self.embeddings)
        elif self.embedding_file.exists():
            self.embedding_file.unlink()

        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.faiss_file))
        elif self.faiss_file.exists():
            self.faiss_file.unlink()

    def load(self) -> None:
        """Load persisted data; rebuild generated artifacts if needed/corrupt."""
        self.chunks = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.doc_hashes = {}
        self.faiss_index = None
        self.bm25 = None

        if self.meta_file.exists():
            try:
                data = json.loads(self.meta_file.read_text(encoding="utf-8"))
                self.chunks = [Chunk(**item) for item in data]
            except Exception:
                self.chunks = []

        if self.doc_hash_file.exists():
            try:
                self.doc_hashes = json.loads(self.doc_hash_file.read_text(encoding="utf-8"))
            except Exception:
                self.doc_hashes = {}

        need_rebuild_embeddings = False
        if self.embedding_file.exists():
            try:
                self.embeddings = np.load(self.embedding_file).astype(np.float32)
                if len(self.embeddings) != len(self.chunks):
                    need_rebuild_embeddings = True
            except Exception:
                need_rebuild_embeddings = True
        elif self.chunks:
            need_rebuild_embeddings = True

        if need_rebuild_embeddings and self.chunks:
            self.embeddings = self.embed_fn([chunk.text for chunk in self.chunks]).astype(np.float32)

        if self.chunks and self.embeddings.size == 0:
            self.embeddings = self.embed_fn([chunk.text for chunk in self.chunks]).astype(np.float32)

        self._rebuild_indexes(prefer_saved_faiss=True)

    def _rebuild_indexes(self, prefer_saved_faiss: bool = False) -> None:
        """Rebuild FAISS and BM25 indexes from in-memory chunks/embeddings."""
        if not self.chunks:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.faiss_index = None
            self.bm25 = None
            return

        if self.embeddings.size == 0 or len(self.embeddings) != len(self.chunks):
            self.embeddings = self.embed_fn([chunk.text for chunk in self.chunks]).astype(np.float32)

        corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        loaded = False
        if prefer_saved_faiss and self.faiss_file.exists():
            try:
                idx = faiss.read_index(str(self.faiss_file))
                if idx.ntotal == len(self.chunks):
                    self.faiss_index = idx
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            dim = int(self.embeddings.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings)
            self.faiss_index = index
