# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HireFlow is an AI-powered recruitment platform that matches candidates to job descriptions using semantic search. It combines BM25 keyword search with Pinecone vector search, routed and re-ranked by Google Gemini.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root (see `.env.example`):
```
GOOGLE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=hireflow   # optional, default: hireflow
LLM_MODEL=gemini-2.5-flash     # optional
```

## Running the Application

```bash
# Web UI (Streamlit)
cd streamlit && streamlit run app.py

# CLI interactive demo
python main.py
```

There is no automated test suite in this repository.

## Architecture

### Data Flow

```
PDFs → ingestion.py → LangChain Documents
     → HybridIndexer → BM25 (in-memory) + VectorStore (Pinecone)
Query → SearchRouter (RunnableBranch)
     → shallow: VectorStore.search_resumes()
     → deep:    HybridIndexer.search_resumes() → LLM rerank
Results → ReRanker.evaluate_candidate_fit() → CandidateEvaluation
        → MemoryRAG.record_search()
```

### Core Components (`core/`)

| File | Class | Role |
|------|-------|------|
| `ingestion.py` | `load_resumes`, `DocumentProcessor` | PDF → LangChain `Document` with metadata |
| `parsing.py` | `ResumeParser`, `JobParser` | Gemini LLM extracts structured `Resume`/`JobDescription` from raw text |
| `vector_store.py` | `VectorStore` | Wraps Pinecone; embeds with Google GenAI (768-dim, cosine); creates `aws us-east-1` ServerlessSpec index on first run |
| `hybrid_indexer.py` | `HybridIndexer` | Builds BM25Okapi over in-memory resume/JD texts; merges BM25 + vector scores by concatenation+sort |
| `search_router.py` | `SearchRouter` | `RunnableBranch` routes to `shallow_search` (vector only) or `deep_search` (hybrid + LLM rerank); auto-mode uses LLM to classify query complexity, falls back to word-count heuristic |
| `re_ranker.py` | `ReRanker` | Gemini LLM scores candidate fit, returns `CandidateEvaluation` |
| `evaluator.py` | `RAGEvaluator` | RAGAS metrics (answer relevancy, context precision, faithfulness, correctness) |
| `memory_rag.py` | `MemoryRAG` | `ConversationBufferMemory` (from `langchain-classic`) tracks search queries and candidate views in-session |

### Utilities (`utils/`)

- **`config.py`** — `AppConfig` (pydantic-settings); also exports flat module-level constants (`GOOGLE_API_KEY`, `LLM_MODEL`, etc.) imported throughout the codebase.
- **`schemas.py`** — Pydantic models: `Resume`, `JobDescription`, `CandidateEvaluation`.
- **`multi_query.py`** — `MultiQueryGenerator` expands a single query into N variants via LLM.
- **`utils.py`** — `get_logger`, `get_embeddings`, `load_pdf`, `is_quota_error` helpers.

### UI (`streamlit/app.py`)

Inserts the project root into `sys.path` at startup so imports from `core/` and `utils/` work when launched from the `streamlit/` subdirectory. Uses Streamlit session state to cache the `SystemManager` instance across rerenders.

## Key Implementation Notes

- **Pinecone index** is auto-created (or recreated on dimension mismatch) inside `VectorStore.ensure_index()` with `ServerlessSpec(cloud="aws", region="us-east-1")`.
- **`data/jds/`** directory is required by `main.py` but is not committed — create it and add job description PDFs before running the CLI demo.
- **Hybrid scoring** in `HybridIndexer.combine_results` concatenates BM25 and vector result lists then sorts by `combined_score`; it does not implement true Reciprocal Rank Fusion despite some field names (`rrf_score`) suggesting it.
- **`langchain-classic`** (not `langchain-community`) provides `ConversationBufferMemory` — this is a separate pinned package.
- All LLM calls use hardcoded inline prompts; there is no separate prompts directory despite the README architecture diagram showing one.
