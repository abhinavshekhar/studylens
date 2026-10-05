# StudyLens (Phase 1)

StudyLens Phase 1 implements the core local retrieval foundation for a future Streamlit RAG assistant.

## Current Scope

Included in this milestone:
- PDF/TXT ingestion with page-aware extraction
- Text cleaning (whitespace normalization, hyphenated line join, repeated header/footer removal)
- Deterministic chunking with overlap and metadata
- Local embeddings + FAISS vector index + BM25 keyword index
- Hybrid retrieval (RRF) and confidence scoring
- Persistent local index store under `DATA_DIR`

Not included yet:
- Streamlit chat UX
- LLM provider integration
- Quiz generation, highlighting, follow-up rewriting

## Project Structure

- `/app.py` - placeholder app entrypoint
- `/config.py` - env-driven settings and pathlib directories
- `/core/ingestion.py` - extraction, cleaning, chunking
- `/core/embeddings.py` - cached embedding model + normalized vectors
- `/core/index_store.py` - persistent index and upload store
- `/core/retriever.py` - vector/BM25/hybrid retrieval
- `/core/models.py` - shared dataclasses
- `/tests` - unit tests for Phase 1 behavior

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

## Configuration

Key environment variables:
- `DATA_DIR` (default `./data`)
- `EMBED_MODEL` (default `paraphrase-multilingual-MiniLM-L12-v2`)
- `CHUNK_WORDS` (default `150`)
- `CHUNK_OVERLAP` (default `30`)
- `TOP_K` (default `4`)
- `RRF_K` (default `60`)
- `MIN_CONFIDENCE` (default `0.30`)

## Architecture (Phase 1)

1. **Ingestion**: parse pages from `.pdf`/`.txt`, clean text, remove repeated page edges.
2. **Chunking**: split into overlapping chunks while preserving `{doc, page, chunk_id}`.
3. **Indexing**: embed chunks, build FAISS (inner-product) and BM25.
4. **Retrieval**: vector + BM25 search, merge via reciprocal-rank fusion.
5. **Persistence**: save chunk metadata, embeddings, FAISS index, doc hashes, and uploads.
