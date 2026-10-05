"""Embedding helpers for StudyLens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

import config


def _noop_cache(func: Callable[..., Any]) -> Callable[..., Any]:
    return func


def _streamlit_cache_resource() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    try:
        import streamlit as st  # type: ignore

        return st.cache_resource(show_spinner=False)
    except Exception:
        return _noop_cache


@_streamlit_cache_resource()
def _load_model_cached(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_model(model_name: str | None = None):
    """Load and cache the sentence-transformers model."""
    return _load_model_cached(model_name or config.EMBED_MODEL)


def embed(texts: Sequence[str], batch_size: int = 32, model: Any | None = None) -> np.ndarray:
    """Embed texts into float32, L2-normalized vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model_instance = model or get_model()
    vectors = model_instance.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
