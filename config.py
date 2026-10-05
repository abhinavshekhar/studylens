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

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


for directory in (DATA_DIR, UPLOADS_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)
