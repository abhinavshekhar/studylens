# StudyLens

StudyLens is being delivered in six independent, reviewable parts.

## Six-part build roadmap

1. **Usable grounded chat MVP** (this PR): Streamlit UI, uploads, document management, grounded chat, citations, confidence behavior, tests.
2. **Retrieval quality**: query rewriting improvements, hybrid/RRF tuning, cross-document balancing, evaluation tooling.
3. **Study tools**: quiz and flashcard generation with strict JSON validation.
4. **Multilingual explanation experience**: Tamil/Hindi support, same-language answers, style modes.
5. **Evidence experience**: PDF page rendering, highlighting, source viewer, retrieval comparison tab.
6. **Production readiness**: robust errors, OCR warnings, security/privacy checks, deployment + CI hardening.

## Current scope (Part 1)

Included:
- PDF/TXT ingestion, chunking, embeddings, FAISS/BM25 hybrid retrieval (Phase 1 foundation)
- Streamlit grounded chat UI using uploaded/indexed documents
- Configurable LLM adapter (`gemini`, `ollama`, `groq`) via environment variables
- Confidence-gated responses with source snippets and chat history follow-ups
- Deterministic tests for grounded chat pipeline behaviors

Not included yet:
- Quiz/flashcard generation
- Highlighted PDF rendering
- OCR fallback pipeline
- Retrieval comparison mode

## Project structure

- `/app.py` - Streamlit app with upload/index/chat flows
- `/config.py` - env-driven settings
- `/core/ingestion.py` - extraction, cleaning, chunking
- `/core/index_store.py` - persistent indexes/uploads
- `/core/retriever.py` - hybrid retrieval + confidence
- `/core/llm.py` - provider-agnostic generation adapter
- `/core/prompts.py` - grounded/rewrite prompts
- `/core/chat.py` - grounded chat orchestration
- `/tests` - unit tests

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy config template:
   ```bash
   cp .env.example .env
   ```
4. Run tests:
   ```bash
   pytest -q
   ```

## Provider configuration

Set these in your shell or `.env`:
- `LLM_PROVIDER=gemini|ollama|groq`
- `LLM_MODEL=<model-name>`
- `GEMINI_API_KEY` (required for Gemini)
- `GROQ_API_KEY` (required for Groq)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)

## Run the app

```bash
streamlit run app.py
```

## Notes

- Do not commit secrets (`.env` is local-only).
- The assistant is grounded: it is instructed to avoid outside knowledge and to report when answer support is missing.
