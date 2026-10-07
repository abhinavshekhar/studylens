from __future__ import annotations

from core.chat import (
    build_context_sources,
    generate_grounded_answer,
    is_vague_document_question,
    should_rewrite_followup,
    should_rewrite_query,
    truncate_history,
)
from core.models import SearchResult


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def hybrid_search(self, query: str, top_k: int = 4):
        self.queries.append((query, top_k))
        return list(self.results)


def test_truncate_history_keeps_last_turns() -> None:
    history = [{"role": "user", "content": f"m{i}"} for i in range(1, 11)]
    trimmed = truncate_history(history, max_history_turns=3)
    assert [item["content"] for item in trimmed] == ["m5", "m6", "m7", "m8", "m9", "m10"]


def test_should_rewrite_followup_for_short_referential_query() -> None:
    history = [{"role": "user", "content": "Explain photosynthesis"}]
    assert should_rewrite_followup("Can you explain it again?", history)
    assert not should_rewrite_followup(
        "Please provide a detailed explanation of photosynthesis stages and chlorophyll reactions",
        history,
    )


def test_vague_document_questions_trigger_rewrite() -> None:
    assert is_vague_document_question("what is inside this")
    assert is_vague_document_question("what is that")
    assert should_rewrite_query("what is inside this", [])
    assert not is_vague_document_question(
        "Explain the detailed syllabus structure and credit distribution for semester one"
    )


def test_context_numbering_is_deterministic() -> None:
    context, sources = build_context_sources(
        [
            SearchResult(chunk_id="b", doc="bio.txt", page=3, text="Beta", rrf_score=0.5),
            SearchResult(chunk_id="a", doc="chem.txt", page=1, text="Alpha", rrf_score=0.4),
        ]
    )
    assert "[1] bio.txt (page 3)" in context
    assert "[2] chem.txt (page 1)" in context
    assert [source["number"] for source in sources] == [1, 2]


def test_below_threshold_skips_answer_llm_call() -> None:
    retriever = FakeRetriever(
        [SearchResult(chunk_id="x", doc="doc.txt", page=1, text="context", rrf_score=0.2)]
    )
    llm_calls: list[tuple[str, list[dict[str, str]], int]] = []

    def fake_llm(system: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        llm_calls.append((system, messages, max_tokens))
        return "unused"

    result = generate_grounded_answer(
        question="What is mitochondria?",
        history=[],
        retriever=retriever,
        explanation_level="Normal",
        top_k=4,
        min_confidence=0.95,
        llm_generate=fake_llm,
        confidence_fn=lambda _: 0.3,
    )

    assert "cannot find" in result.answer.lower()
    assert result.sources == []
    assert llm_calls == []


def test_source_order_matches_retrieval_rank() -> None:
    retriever = FakeRetriever(
        [
            SearchResult(chunk_id="2", doc="second.txt", page=2, text="second", rrf_score=0.9),
            SearchResult(chunk_id="1", doc="first.txt", page=1, text="first", rrf_score=0.8),
        ]
    )

    def fake_llm(_: str, __: list[dict[str, str]], ___: int) -> str:
        return "Grounded answer [1][2]"

    result = generate_grounded_answer(
        question="Summarize",
        history=[],
        retriever=retriever,
        explanation_level="Simple",
        top_k=2,
        llm_generate=fake_llm,
        confidence_fn=lambda _: 0.9,
    )

    assert [source["document"] for source in result.sources] == ["second.txt", "first.txt"]
    assert [source["number"] for source in result.sources] == [1, 2]


def test_prompt_construction_includes_grounding_constraints() -> None:
    retriever = FakeRetriever(
        [SearchResult(chunk_id="1", doc="notes.txt", page=4, text="ATP stores energy", rrf_score=1.0)]
    )
    captured: dict[str, object] = {}

    def fake_llm(system: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        captured["system"] = system
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return "Answer [1]"

    generate_grounded_answer(
        question="What does ATP do?",
        history=[],
        retriever=retriever,
        explanation_level="Exam",
        top_k=1,
        llm_generate=fake_llm,
        confidence_fn=lambda _: 0.8,
    )

    system = str(captured["system"])
    user_prompt = str((captured["messages"])[0]["content"])

    assert "Do not use outside knowledge" in system
    assert "Respond in the same language" in system
    assert "exam-focused detail" in system.lower()
    assert "[1] notes.txt (page 4)" in user_prompt
