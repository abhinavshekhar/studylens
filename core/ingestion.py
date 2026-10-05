"""Document ingestion utilities: extraction, cleaning, and chunking."""

from __future__ import annotations

import io
import math
import re
from collections import Counter
from typing import Iterable

from pypdf import PdfReader

from core.models import Chunk


def extract_pages(filename: str, file_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from PDF/TXT bytes."""
    lower = filename.lower()
    if lower.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
        return [(1, text)]
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[tuple[int, str]] = []
        for page_no, page in enumerate(reader.pages, start=1):
            pages.append((page_no, page.extract_text() or ""))
        return pages
    raise ValueError(f"Unsupported file type for '{filename}'. Only .pdf and .txt are supported.")


def clean_text(text: str) -> str:
    """Normalize whitespace and stitch hyphenated line breaks."""
    stitched = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    stitched = stitched.replace("\r\n", "\n").replace("\r", "\n")
    collapsed = re.sub(r"[\t\n\f\v]+", " ", stitched)
    collapsed = re.sub(r"\s{2,}", " ", collapsed)
    return collapsed.strip()


def remove_repeated_headers_footers(
    pages: Iterable[tuple[int, str]], threshold: float = 0.6
) -> list[tuple[int, str]]:
    """Conservatively remove repeated first/last lines across pages."""
    page_list = list(pages)
    normalized_lines: list[tuple[int, list[str]]] = []
    for page_no, text in page_list:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalized_lines.append((page_no, lines))

    non_empty_pages = [lines for _, lines in normalized_lines if lines]
    total = len(non_empty_pages)
    if total < 3:
        return page_list

    needed = max(2, math.ceil(total * threshold))
    header_counts = Counter(lines[0] for lines in non_empty_pages)
    footer_counts = Counter(lines[-1] for lines in non_empty_pages)

    repeated_headers = {
        line for line, count in header_counts.items() if count >= needed and len(line) <= 120
    }
    repeated_footers = {
        line for line, count in footer_counts.items() if count >= needed and len(line) <= 120
    }

    cleaned: list[tuple[int, str]] = []
    for page_no, text in page_list:
        lines = [line for line in text.splitlines()]
        stripped = [line.strip() for line in lines if line.strip()]
        header = stripped[0] if stripped else None
        footer = stripped[-1] if stripped else None

        start = 0
        end = len(lines)
        if header and header in repeated_headers:
            while start < end and not lines[start].strip():
                start += 1
            if start < end:
                start += 1
        if footer and footer in repeated_footers:
            while end > start and not lines[end - 1].strip():
                end -= 1
            if end > start:
                end -= 1
        cleaned.append((page_no, "\n".join(lines[start:end])))
    return cleaned


def is_scanned_or_empty_pdf(pages: Iterable[tuple[int, str]]) -> bool:
    """Return True if a PDF appears to have no extractable text."""
    return all(not clean_text(text) for _, text in pages)


def chunk_pages(
    doc: str,
    pages: Iterable[tuple[int, str]],
    size: int = 150,
    overlap: int = 30,
) -> list[Chunk]:
    """Split pages into deterministic overlapping word chunks."""
    if size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    if overlap < 0:
        raise ValueError("Chunk overlap must be non-negative.")
    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    step = size - overlap
    chunks: list[Chunk] = []
    for page_no, raw_text in pages:
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        words = cleaned.split()
        for start in range(0, len(words), step):
            piece = words[start : start + size]
            if not piece:
                continue
            text = " ".join(piece).strip()
            if not text:
                continue
            chunk_id = f"{doc}::p{page_no}::w{start}"
            chunks.append(Chunk(chunk_id=chunk_id, doc=doc, page=page_no, text=text))
    return chunks
