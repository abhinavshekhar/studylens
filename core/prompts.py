"""Prompt builders for grounded StudyLens chat flows."""

from __future__ import annotations

EXPLANATION_STYLES: dict[str, str] = {
    "Simple": "Use very simple language and short sentences.",
    "Normal": "Use clear, concise explanations for a typical student.",
    "Exam": "Use exam-focused detail, definitions, and key takeaways.",
}

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's latest question into a standalone query using only chat history context. "
    "Do not answer the question. Return only the rewritten standalone query."
)


def build_grounded_system_prompt(explanation_level: str) -> str:
    """Create the system prompt for grounded answering with style guidance."""
    style_instruction = EXPLANATION_STYLES.get(explanation_level, EXPLANATION_STYLES["Normal"])
    return (
        "You are StudyLens, a grounded study assistant. "
        "Answer strictly from the provided context snippets only. "
        "If the context does not contain enough evidence, say you cannot find the answer in the uploaded documents. "
        "Do not use outside knowledge. "
        "When you cite evidence, refer to snippet numbers like [1], [2]. "
        "Respond in the same language as the user's question. "
        f"{style_instruction}"
    )


def build_grounded_user_prompt(question: str, standalone_query: str, context_block: str) -> str:
    """Create the user prompt containing the query and retrieved evidence snippets."""
    return (
        f"Question: {question}\n"
        f"Standalone query used for retrieval: {standalone_query}\n\n"
        "Context snippets:\n"
        f"{context_block}\n\n"
        "Answer using only the context above. If unsupported, clearly say you cannot find the answer in the uploaded documents."
    )
