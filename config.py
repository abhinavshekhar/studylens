"""Runtime configuration for StudyLens."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).expanduser().resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "150"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.30"))
RRF_K = int(os.getenv("RRF_K", "60"))


for directory in (DATA_DIR, UPLOADS_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)
