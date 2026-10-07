"""Streamlit application for StudyLens grounded chat MVP."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from core.chat import ChatResult, generate_grounded_answer
from core.index_store import IndexStore
from core.retriever import Retriever


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "High"
    if value >= 0.45:
        return "Medium"
    return "Low"


@st.cache_resource(show_spinner=False)
def _build_store(data_dir: str) -> IndexStore:
    store = IndexStore(data_dir=Path(data_dir))
    store.load()
    return store


def _render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])
    confidence = float(message.get("confidence", 0.0))
    st.caption(f"Confidence: {_confidence_label(confidence)} ({confidence:.2f})")
    sources = message.get("sources", [])
    if sources:
        with st.expander("Sources"):
            for source in sources:
                st.markdown(
                    f"**[{source['number']}] {source['document']}** · page {source['page']} · score {source['score']}"
                )
                st.write(source["snippet"])


def _process_uploads(store: IndexStore, files: list) -> None:
    for file in files:
        try:
            with st.spinner(f"Indexing {file.name}..."):
                changed = store.add_document(file.name, file.getvalue())
            if changed:
                st.sidebar.success(f"Indexed: {file.name}")
            else:
                st.sidebar.info(f"No changes detected in: {file.name}")
        except Exception as exc:
            st.sidebar.error(f"Could not index {file.name}: {exc}")


def _remove_document(store: IndexStore, name: str) -> None:
    try:
        removed = store.remove_document(name)
        if removed:
            st.sidebar.success(f"Removed: {name}")
        else:
            st.sidebar.info(f"Document not found: {name}")
    except Exception as exc:
        st.sidebar.error(f"Could not remove {name}: {exc}")


def _reset_knowledge_base(store: IndexStore) -> None:
    for doc in list(store.list_documents()):
        store.remove_document(str(doc["name"]))


def _llm_configured() -> bool:
    provider = config.LLM_PROVIDER.strip().lower()
    if provider == "gemini":
        return bool(config.GEMINI_API_KEY.strip())
    if provider == "groq":
        return bool(config.GROQ_API_KEY.strip())
    if provider == "ollama":
        return True
    return False


def main() -> None:
    """Render the StudyLens grounded chat experience."""
    st.set_page_config(page_title="StudyLens", page_icon="📚", layout="wide")
    config.refresh_settings()

    st.title("📚 StudyLens")
    st.caption("Grounded chat over your uploaded study documents")
    st.caption(f"LLM: {config.LLM_PROVIDER} · {config.LLM_MODEL}")

    if not _llm_configured():
        st.warning(
            "Cloud LLM is not configured. Chat answers are disabled until an API key is added. "
            "Document upload and search still work."
        )
        with st.expander("How to enable chat on Streamlit Cloud"):
            st.markdown(
                "1. Open your app on [share.streamlit.io](https://share.streamlit.io)\n"
                "2. Click **Manage app** (bottom right) → **Settings** → **Secrets**\n"
                "3. Paste this and save (then reboot the app if prompted):\n"
            )
            st.code(
                'LLM_PROVIDER = "gemini"\n'
                'LLM_MODEL = "gemini-3.5-flash-lite"\n'
                'GEMINI_API_KEY = "paste-your-key-here"',
                language="toml",
            )
            st.caption(
                "Get a free key at https://aistudio.google.com/apikey. "
                "Cursor Cloud Agent secrets are separate and do not apply here."
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    store = _build_store(str(config.DATA_DIR))
    retriever = Retriever(store)

    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDF/TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.caption("Selected files are not searchable until you click **Process uploads**.")
        if st.button("Process uploads", use_container_width=True):
            if uploaded_files:
                _process_uploads(store, uploaded_files)
            else:
                st.info("Select one or more PDF/TXT files first.")

        st.subheader("Indexed documents")
        docs = store.list_documents()
        if docs:
            remove_enabled = st.checkbox("Enable remove controls", value=False)
            for doc in docs:
                cols = st.columns([4, 1])
                cols[0].write(f"{doc['name']} ({doc['chunks']} chunks)")
                if cols[1].button("🗑️", key=f"rm::{doc['name']}", disabled=not remove_enabled):
                    _remove_document(store, str(doc["name"]))
                    st.rerun()
        else:
            st.caption("No indexed documents yet.")

        explanation_level = st.selectbox("Explanation level", ["Simple", "Normal", "Exam"], index=1)
        top_k = st.slider("Top-k sources", min_value=1, max_value=10, value=config.TOP_K)

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Reset knowledge base", use_container_width=True):
            with st.spinner("Removing indexed documents..."):
                _reset_knowledge_base(store)
            st.sidebar.success("Knowledge base reset.")
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
            if msg["role"] in {"user", "assistant"}
        ][:-1]

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and composing grounded answer..."):
                    result: ChatResult = generate_grounded_answer(
                        question=question,
                        history=history,
                        retriever=retriever,
                        explanation_level=explanation_level,
                        top_k=top_k,
                    )
                assistant_message = {
                    "role": "assistant",
                    "content": result.answer,
                    "confidence": result.confidence,
                    "sources": result.sources,
                    "standalone_query": result.standalone_query,
                }
                _render_assistant_message(assistant_message)
                st.session_state.messages.append(assistant_message)
            except Exception as exc:
                friendly = "Sorry, something went wrong while generating an answer."
                st.error(f"{friendly} {exc}")


if __name__ == "__main__":
    main()
