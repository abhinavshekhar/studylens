"""Grounded chat pipeline for StudyLens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import config
from core import llm
from core.models import SearchResult
from core.prompts import REWRITE_SYSTEM_PROMPT, build_grounded_system_prompt, build_grounded_user_prompt
from core.retriever import Retriever, confidence

REFERENTIAL_TOKENS = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "he",
    "she",
    "again",
    "more",
    "above",
    "previous",
    "earlier",
    "same",
    "its",
}


@dataclass(frozen=True)
class ChatResult:
    """Result payload returned by grounded chat inference."""

    answer: str
    sources: list[dict[str, str | int | float]]
    confidence: float
    standalone_query: str


def truncate_history(history: Sequence[dict[str, str]], max_history_turns: int) -> list[dict[str, str]]:
    """Keep only the last N user+assistant turns (2N messages)."""
    max_messages = max(0, max_history_turns * 2)
    if max_messages == 0:
        return []
    return list(history[-max_messages:])


def should_rewrite_followup(question: str, history: Sequence[dict[str, str]]) -> bool:
    """Decide whether a query should be rewritten as a standalone question."""
    if not history:
        return False
    words = re.findall(r"\w+", question.lower())
    if len(words) > 12:
        return False
    return any(word in REFERENTIAL_TOKENS for word in words)


def build_context_sources(results: Sequence[SearchResult]) -> tuple[str, list[dict[str, str | int | float]]]:
    """Build numbered context snippets and structured source metadata."""
    snippets: list[str] = []
    sources: list[dict[str, str | int | float]] = []
    for idx, item in enumerate(results, start=1):
        snippet = " ".join(item.text.split())
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."
        score = round(float(item.rrf_score or item.vector_score or item.bm25_score or 0.0), 4)
        snippets.append(f"[{idx}] {item.doc} (page {item.page}) score={score}: {snippet}")
        sources.append(
            {
                "number": idx,
                "document": item.doc,
                "page": item.page,
                "snippet": snippet,
                "score": score,
            }
        )
    return "\n\n".join(snippets), sources


def generate_grounded_answer(
    question: str,
    history: Sequence[dict[str, str]],
    retriever: Retriever,
    explanation_level: str,
    top_k: int,
    max_tokens: int = config.LLM_MAX_TOKENS,
    max_history_turns: int = config.MAX_HISTORY_TURNS,
    min_confidence: float = config.MIN_CONFIDENCE,
    llm_generate: Callable[[str, Sequence[dict[str, str]], int], str] = llm.generate,
    confidence_fn: Callable[[list[SearchResult]], float] = confidence,
) -> ChatResult:
    """Generate a grounded response with deterministic source construction."""
    trimmed_history = truncate_history(history, max_history_turns)

    standalone_query = question
    if should_rewrite_followup(question, trimmed_history):
        rewrite_messages = [
            {
                "role": "user",
                "content": (
                    "Conversation history:\n"
                    + "\n".join([f"{m['role']}: {m['content']}" for m in trimmed_history])
                    + f"\n\nLatest question: {question}\n"
                    "Return only the rewritten standalone query."
                ),
            }
        ]
        rewritten = llm_generate(REWRITE_SYSTEM_PROMPT, rewrite_messages, 96).strip()
        if rewritten:
            standalone_query = rewritten

    results = retriever.hybrid_search(standalone_query, top_k=top_k)
    conf = confidence_fn(results)
    if not results or conf < min_confidence:
        return ChatResult(
            answer=(
                "I cannot find the answer in the uploaded documents with enough confidence. "
                "Please upload more relevant material or ask a more specific question."
            ),
            sources=[],
            confidence=conf,
            standalone_query=standalone_query,
        )

    context_block, sources = build_context_sources(results)
    system_prompt = build_grounded_system_prompt(explanation_level)
    user_prompt = build_grounded_user_prompt(question, standalone_query, context_block)
    answer = llm_generate(system_prompt, [{"role": "user", "content": user_prompt}], max_tokens).strip()

    return ChatResult(
        answer=answer or "I cannot find the answer in the uploaded documents.",
        sources=sources,
        confidence=conf,
        standalone_query=standalone_query,
    )
