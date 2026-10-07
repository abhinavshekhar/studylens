"""Runtime configuration for StudyLens."""

from __future__ import annotations

import os
from pathlib import Path


def _setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value is not None and value != "":
        return value

    try:
        import streamlit as st  # type: ignore

        if name in st.secrets:
            secret = st.secrets[name]
            return str(secret) if secret is not None else default
    except Exception:
        pass

    return default


BASE_DIR = Path(__file__).resolve().parent


def refresh_settings() -> None:
    """Reload settings on each Streamlit run so secrets are picked up."""
    global DATA_DIR, UPLOADS_DIR, INDEX_DIR
    global EMBED_MODEL, CHUNK_WORDS, CHUNK_OVERLAP, TOP_K, MIN_CONFIDENCE, RRF_K
    global LLM_PROVIDER, LLM_MODEL, LLM_MAX_TOKENS, MAX_HISTORY_TURNS
    global GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_BASE_URL

    DATA_DIR = Path(_setting("DATA_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
    UPLOADS_DIR = DATA_DIR / "uploads"
    INDEX_DIR = DATA_DIR / "index"

    EMBED_MODEL = _setting("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    CHUNK_WORDS = int(_setting("CHUNK_WORDS", "150"))
    CHUNK_OVERLAP = int(_setting("CHUNK_OVERLAP", "30"))
    TOP_K = int(_setting("TOP_K", "4"))
    MIN_CONFIDENCE = float(_setting("MIN_CONFIDENCE", "0.22"))
    RRF_K = int(_setting("RRF_K", "60"))

    LLM_PROVIDER = _setting("LLM_PROVIDER", "gemini")
    LLM_MODEL = _setting("LLM_MODEL", "gemini-3.5-flash-lite")
    LLM_MAX_TOKENS = int(_setting("LLM_MAX_TOKENS", "512"))
    MAX_HISTORY_TURNS = int(_setting("MAX_HISTORY_TURNS", "6"))
    GEMINI_API_KEY = _setting("GEMINI_API_KEY", "")
    GROQ_API_KEY = _setting("GROQ_API_KEY", "")
    OLLAMA_BASE_URL = _setting("OLLAMA_BASE_URL", "http://localhost:11434")

    for directory in (DATA_DIR, UPLOADS_DIR, INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)


refresh_settings()
