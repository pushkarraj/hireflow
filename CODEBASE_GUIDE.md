# HireFlow — Complete Codebase Guide

> An AI-powered recruitment platform that matches candidates to job descriptions using semantic search, keyword search, and LLM re-ranking.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [End-to-End Data Flow](#end-to-end-data-flow)
3. [Module Reference](#module-reference)
   - [utils/config.py](#utilsconfigpy)
   - [utils/schemas.py](#utilsschemaspy)
   - [utils/utils.py](#utilsutilspy)
   - [utils/multi_query.py](#utilsmulti_querypy)
   - [core/ingestion.py](#coreingestionpy)
   - [core/parsing.py](#coreparsingpy)
   - [core/vector_store.py](#corevector_storepy)
   - [core/hybrid_indexer.py](#corehybrid_indexerpy)
   - [core/search_router.py](#coresearch_routerpy)
   - [core/re_ranker.py](#corere_rankerpy)
   - [core/memory_rag.py](#corememory_ragpy)
   - [core/evaluator.py](#coreevaluatorpy)
   - [streamlit/app.py](#streamlitapppy)
   - [main.py](#mainpy)
4. [Module Dependency Map](#module-dependency-map)
5. [Data Models](#data-models)
6. [Search Pipeline Deep-Dive](#search-pipeline-deep-dive)
7. [Configuration & Environment](#configuration--environment)
8. [Key Design Decisions & Gotchas](#key-design-decisions--gotchas)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Entry Points                                │
│              streamlit/app.py          main.py                       │
└───────────────────┬─────────────────────────┬───────────────────────┘
                    │                         │
                    ▼                         ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Core Modules                                │
│                                                                   │
│  ingestion.py ──► parsing.py                                      │
│       │                                                           │
│       ▼                                                           │
│  hybrid_indexer.py ◄──── vector_store.py (Pinecone)               │
│       │                        ▲                                  │
│       └──────► search_router.py (RunnableBranch)                  │
│                       │                                           │
│                       ▼                                           │
│                  re_ranker.py ──► CandidateEvaluation             │
│                                                                   │
│  memory_rag.py  (tracks every search + view)                      │
│  evaluator.py   (RAGAS quality metrics)                           │
└───────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Utils / Support                             │
│   config.py   schemas.py   utils.py   multi_query.py              │
└───────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Data Flow

### Phase 1 — Ingestion & Indexing

```
PDF files (data/resumes/, data/jds/)
    │
    ▼  load_pdf() in utils/utils.py (uses LangChain PyPDFLoader)
    │
    ▼  load_resumes() / load_job_descriptions() in core/ingestion.py
    │  Creates LangChain Document objects with metadata:
    │    resume → {candidate_id, name, filename, source}
    │    jd     → {jd_id, title, filename, source}
    │
    ▼  HybridIndexer.index_resumes() / index_job_descriptions()
    │
    ├──► BM25Okapi (in-memory) — tokenizes and builds keyword index
    │
    └──► VectorStore.add_resumes() — embeds with Google GenAI
             (model: gemini-embedding-001, 3072 dims, cosine)
             → upserts vectors + metadata to Pinecone (AWS us-east-1)
```

### Phase 2 — Search

```
User query (text string)
    │
    ▼  SearchRouter.search()
    │
    ├── route_search() decides:
    │     • LLM classification (Gemini) → "shallow" or "deep"
    │     • Fallback: word count ≤ 3 → shallow, else deep
    │
    ├── SHALLOW path: VectorStore.search_resumes()
    │     → embed query → Pinecone cosine similarity → top-K
    │
    └── DEEP path: HybridIndexer.search_resumes()
          │
          ├── BM25 scores (normalized 0-1)
          ├── Pinecone vector scores (cosine 0-1)
          ├── combine_results() → concatenate + sort by combined_score
          │
          └── llm_rerank() (optional): Gemini re-orders by semantic fit
```

### Phase 3 — Evaluation & Memory

```
Search results
    │
    ├──► ReRanker.evaluate_candidate()
    │      Gemini prompt → strengths / gaps / risks / fit_score (0-100)
    │      Returns CandidateEvaluation Pydantic model
    │
    ├──► MemoryRAG.record_search()
    │      Stores query + result count in ConversationBufferMemory
    │
    └──► RAGEvaluator.evaluate_search_quality()
           RAGAS metrics: answer_relevancy, context_precision,
           faithfulness, answer_correctness → overall_score
```

---

## Module Reference

---

### `utils/config.py`

**Role:** Single source of truth for all configuration values.

**How it works:**
- Uses `pydantic-settings` `BaseSettings` to load from environment / `.env` file.
- Exports flat module-level constants (`GOOGLE_API_KEY`, `LLM_MODEL`, etc.) so every module can do `from utils.config import GOOGLE_API_KEY` without instantiating anything.

**Key exports:**

| Constant | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | `""` | Google Gemini API key |
| `LLM_MODEL` | `gemini-2.5-flash` | Model used for all LLM calls |
| `PINECONE_API_KEY` | `""` | Pinecone API key |
| `PINECONE_INDEX_NAME` | `hireflow` | Pinecone index name |
| `PINECONE_DIMENSION` | `3072` | Embedding dimension |
| `PINECONE_METRIC` | `cosine` | Similarity metric |
| `MAX_TEXT_LENGTH` | `4000` | Max chars for text processing |

**Connected to:** Everything — imported by all core modules and utilities.

---

### `utils/schemas.py`

**Role:** Defines the three core Pydantic data models used throughout the system.

**Models:**

```
Resume
  ├── candidate_id: str          (unique, e.g. "c_john_doe")
  ├── name: str
  ├── email / phone / location   (optional)
  ├── text: str                  (raw resume text)
  ├── skills: List[str]
  └── experience: Optional[int]  (years)

JobDescription
  ├── jd_id: str                 (unique, e.g. "jd_senior_accountant")
  ├── title: str
  ├── text: str                  (full JD text)
  ├── required_skills: List[str]
  ├── optional_skills: List[str]
  ├── min_experience: Optional[int]
  └── location: Optional[str]

CandidateEvaluation
  ├── candidate_id: str
  ├── fit_score: int             (0–100)
  ├── strengths: List[str]
  ├── gaps: List[str]
  ├── risks: List[str]
  ├── summary: str
  └── evidence: Optional[dict]
```

**Connected to:** `parsing.py`, `re_ranker.py`, `evaluator.py`, `streamlit/app.py`

---

### `utils/utils.py`

**Role:** Shared helper functions for text processing, PDF loading, embeddings, and filtering.

**Key functions:**

| Function | What it does |
|---|---|
| `get_logger(name)` | Creates a standardized timestamped logger |
| `clean_text(text)` | Strips extra whitespace and non-standard characters |
| `truncate_text(text, max_length)` | Cuts text at max length with ellipsis |
| `load_pdf(file_path)` | Uses `PyPDFLoader` (LangChain) to extract all page text from a PDF |
| `split_text(text, chunk_size, chunk_overlap)` | `RecursiveCharacterTextSplitter` — splits large text into overlapping chunks |
| `is_quota_error(error)` | Detects Google API 429/quota errors for graceful fallback |
| `get_embeddings()` | Creates `GoogleGenerativeAIEmbeddings` (model: `gemini-embedding-001`) |
| `filter_by_skills(candidates, skills)` | Filters candidate list to those with ALL required skills |
| `filter_by_location(candidates, locations)` | Case-insensitive partial location match |
| `filter_by_experience(candidates, min_exp)` | Minimum experience filter |
| `apply_filters(candidates, ...)` | Runs all three filters in sequence |

**Connected to:** `ingestion.py`, `parsing.py`, `vector_store.py`, `hybrid_indexer.py`, `search_router.py`, `re_ranker.py`

---

### `utils/multi_query.py`

**Role:** Expands a single search query into multiple semantically related variants using Gemini LLM.

**Class:** `MultiQueryGenerator`

**How it works:**
1. Sends the original query to Gemini with a prompt asking for N related queries.
2. Parses the response line-by-line to extract clean query strings.
3. Falls back to simple word-truncation rules if LLM is unavailable.

**Example:**
```
Input:  "senior accountant"
Output: ["senior accountant with 5 years experience",
         "CPA certified accountant financial reporting",
         "accounting professional QuickBooks Excel"]
```

**Note:** `MultiQueryGenerator` is defined but is **not wired into the default search pipeline**. It exists as an optional utility that can be called explicitly.

**Connected to:** `utils/config.py`, `utils/utils.py`

---

### `core/ingestion.py`

**Role:** Converts PDF files on disk into LangChain `Document` objects with rich metadata.

**Functions:**

| Function | Input | Output |
|---|---|---|
| `load_resumes(directory)` | Path to folder of PDF resumes | `List[Document]` each with `{candidate_id, name, filename, source}` |
| `load_job_descriptions(directory)` | Path to folder of JD PDFs | `List[Document]` each with `{jd_id, title, filename, source}` |

**`candidate_id`** is derived from the PDF filename (e.g. `john_doe.pdf` → `c_john_doe`).  
**`jd_id`** is similarly derived (e.g. `senior_accountant.pdf` → `jd_senior_accountant`).

**Class:** `DocumentProcessor` — Legacy wrapper; kept for backward compatibility with `streamlit/app.py`. Delegates to `load_pdf()` and `split_text()` from `utils.py`.

**Information flow:**
```
PDF file ──► load_pdf() ──► raw text ──► Document(page_content=text, metadata={...})
```

**Connected to:** `utils/utils.py` (load_pdf), `core/hybrid_indexer.py` (receives Documents), `streamlit/app.py`

---

### `core/parsing.py`

**Role:** Uses Gemini LLM to extract structured fields from raw resume or job description text.

**Classes:**

#### `ResumeParser`
- Builds a LangChain chain: `PromptTemplate | ChatGoogleGenerativeAI | PydanticOutputParser`
- `parse_resume(text, candidate_id)` → returns a `dict` matching the `Resume` schema with extra keys `raw_text` and `parsing_method`.

#### `JobParser`
- Same architecture as `ResumeParser`.
- `parse_job_description(text, jd_id)` → returns a `dict` matching the `JobDescription` schema.

**Processing steps inside each parser:**
```
raw text
  → clean_text()        (remove noise)
  → truncate_text(8000) (cap at 8000 chars)
  → PromptTemplate      (inject text + format instructions)
  → Gemini LLM          (temperature=0.1 for consistency)
  → PydanticOutputParser (validate JSON → Resume/JobDescription)
  → .model_dump()        (Pydantic → dict)
  → add metadata (candidate_id, raw_text, parsing_method)
```

**Connected to:** `utils/config.py`, `utils/schemas.py`, `utils/utils.py`, `streamlit/app.py`, `main.py`

---

### `core/vector_store.py`

**Role:** Manages the Pinecone vector database — creation, embedding, upsert, and similarity search.

**Class:** `VectorStore`

**Initialization flow:**
```
VectorStore()
  └── initialize()
        ├── Pinecone(api_key=...) → self.client
        ├── ensure_index()
        │     • list existing indexes
        │     • if dimension mismatch → delete + recreate
        │     • create ServerlessSpec(cloud="aws", region="us-east-1")
        ├── get_embeddings() → self.embeddings (Google GenAI 3072-dim)
        └── client.Index(name) → self.index
```

**Key methods:**

| Method | Description |
|---|---|
| `add_resumes(docs)` | Embeds each `Document`, upserts with id `resume_{i}_{candidate_id}`, type=`resume` |
| `add_job_descriptions(docs)` | Same for JDs, type=`job_description` |
| `search_resumes(query, top_k, filters)` | Embeds query → Pinecone cosine query → returns list of `{page_content, metadata, score}` |
| `search_job_descriptions(query, top_k, filters)` | Same for JDs |
| `convert_filters_to_pinecone(filters)` | Translates Python dicts to Pinecone filter format (`$eq`, `$in`, `$gte`, `$lte`) |
| `get_stats()` | Returns total vector count, dimension, metric |

**Connected to:** `utils/config.py`, `utils/utils.py` (get_embeddings, is_quota_error), `core/hybrid_indexer.py`

---

### `core/hybrid_indexer.py`

**Role:** Combines BM25 in-memory keyword search with Pinecone vector search. This is the **primary search backend** for deep search queries.

**Class:** `HybridIndexer`

**State:**
```
bm25_resumes    BM25Okapi   — keyword index over resume texts
bm25_jds        BM25Okapi   — keyword index over JD texts
resume_texts    List[str]   — lowercase resume texts (corpus for BM25)
resume_metadata List[dict]  — original metadata per resume (name, candidate_id, etc.)
jd_texts        List[str]   — lowercase JD texts
vector_store    VectorStore — shared Pinecone instance
```

**Indexing flow:**
```
index_resumes(docs)
  ├── Append each doc's lowercased text to resume_texts
  ├── Append doc.metadata to resume_metadata
  ├── Rebuild BM25Okapi(tokenized_texts) over full corpus
  └── vector_store.add_resumes(docs)  →  Pinecone upsert
```

**Search flow (`search_resumes`):**
```
query
  ├── BM25: query.lower().split() → bm25.get_scores() → sorted top-K
  └── Vector: vector_store.search_resumes(query, top_k*2, {type:resume})
                                             │
                                             ▼
                                    combine_results()
                                      • Normalize BM25 scores to [0,1]
                                      • Concatenate BM25 + vector result lists
                                      • Sort descending by combined_score
                                      • Return top_k
```

**Important:** Scoring is a simple concatenation + sort — **not** true Reciprocal Rank Fusion despite the field name `rrf_score` appearing in some places.

**Connected to:** `core/vector_store.py`, `core/search_router.py`, `utils/utils.py`

---

### `core/search_router.py`

**Role:** Intelligently routes a query to either fast (shallow/vector-only) or comprehensive (deep/hybrid+LLM) search using LangChain `RunnableBranch`.

**Class:** `SearchRouter`

**Internal components:**
- `hybrid_indexer` — `HybridIndexer` instance (for deep search)
- `vector_store` — `VectorStore` instance (for shallow search)
- `llm` — Gemini (for routing decisions and re-ranking)
- `search_chain` — `RunnableBranch` built at init time

**Routing logic (`route_search`):**
```
search_mode == "shallow" → "shallow"
search_mode == "deep"    → "deep"
search_mode == "auto":
    if LLM available:
        Ask Gemini: shallow or deep?
    else (fallback):
        word count ≤ 3 → shallow
        word count > 3 → deep
```

**Shallow search path:**
```
vector_store.search_resumes(query, top_k, filters)
  → list of {candidate_id, name, score, metadata, search_type="shallow_vector"}
```

**Deep search path:**
```
hybrid_indexer.search_resumes(query, top_k)
  → (optional) llm_rerank(results, query, jd_context)
      Gemini prompt: rank candidates 1..N → parse order → reorder list
  → list of {candidate_id, name, score, metadata, search_type="deep_hybrid_reranked"}
```

**Public method:**
```python
SearchRouter.search(query, top_k=5, filters=None, search_mode="auto", jd_context="")
  → {"results": [...], "search_type": "...", "query": ..., "top_k": ...}
```

**Connected to:** `core/hybrid_indexer.py`, `core/vector_store.py`, `utils/config.py`, `utils/utils.py`

---

### `core/re_ranker.py`

**Role:** Uses Gemini LLM to perform deep, qualitative evaluation of a candidate's fit for a job.

**Class:** `ReRanker`

**`evaluate_candidate(resume: Resume, jd: JobDescription) → CandidateEvaluation`**

LLM prompt asks for:
1. 3 key strengths
2. 3 areas for improvement
3. Any risks
4. A brief summary

Response is parsed via `extract_section()` and `extract_summary()` helpers.

`fit_score` formula:
```python
fit_score = min(100, max(0, 50 + len(strengths) * 15 - len(gaps) * 10))
```

**Fallback (`simple_evaluation`):** Rule-based scoring when LLM is unavailable — checks experience threshold and required skill overlap.

**`re_rank_candidates(candidates, jd)` → `List[CandidateEvaluation]`**
- Accepts raw search result dicts, converts to `Resume` objects, calls `evaluate_candidate()` for each, sorts by `fit_score` descending.

**Connected to:** `utils/config.py`, `utils/schemas.py`, `utils/utils.py`, `streamlit/app.py`, `main.py`

---

### `core/memory_rag.py`

**Role:** Maintains an in-session conversation log of search queries and candidate interactions using LangChain memory.

**Class:** `MemoryRAG`

Uses `ConversationBufferMemory` (from `langchain-classic` package, **not** `langchain-community`).

**Methods:**

| Method | Stores |
|---|---|
| `record_search(query, results_count)` | `HumanMessage("Search: {query}")` + `AIMessage("Found N results")` |
| `record_candidate_view(candidate_name)` | `HumanMessage("Viewed candidate: {name}")` + `AIMessage(...)` |
| `get_search_history()` | Returns last 5 search queries |
| `get_memory_stats()` | `{total_messages, search_count, candidate_views}` |

**Important:** Memory is in-session only — it is not persisted to disk or a database. All history is lost when the app restarts.

**Connected to:** `streamlit/app.py`, `main.py`

---

### `core/evaluator.py`

**Role:** Measures the quality of search results using the RAGAS framework.

**Class:** `RAGEvaluator`

**Metrics used:**

| Metric | Weight | Meaning |
|---|---|---|
| `answer_relevancy` | 30% | How relevant candidate profiles are to the query |
| `context_precision` | 30% | Precision of retrieved context |
| `faithfulness` | 20% | Factual consistency of the answer with context |
| `answer_correctness` | 20% | Overall answer quality |

**Evaluation flow:**
```
evaluate_search_quality(query, expected_skills, search_mode, top_k)
  → SearchRouter().search(query, top_k)
  → prepare_ragas_data()   — builds a pandas DataFrame:
      {question, contexts: [candidate_metadata], ground_truth, answer}
  → ragas.evaluate(dataset, metrics)
  → calculate_overall_score() (weighted average)
  → store_evaluation() (in-memory history)
  → returns RAGEvaluationMetrics dataclass
```

**Ground truth generation** (`create_ground_truth`):
- Compares candidate skills vs. expected skills
- Returns label: "Excellent/Good/Moderate/Poor match: N/M skills"

**Connected to:** `core/search_router.py`, `utils/utils.py`

---

### `streamlit/app.py`

**Role:** Web UI entry point. Renders the interactive application and orchestrates all core modules.

**Key classes:**

#### `SystemManager`
- Initializes all core components on first call to `initialize()`.
- Stores components in a dict and provides `get_component(name)` accessor.
- Auto-indexes resumes from `data/resumes/` at startup.
- Passes the already-initialized `VectorStore` into `HybridIndexer` to avoid opening a duplicate Pinecone connection.

#### `HireFlowUI`
- Pure UI class — delegates all business logic to `SystemManager` and core modules.
- `render_upload_section()` — PDF uploader → process → `hybrid_indexer.index_resumes()`
- `render_search_section()` — search form → `hybrid_indexer.search_resumes()` → `re_ranker.re_rank_candidates()`
- `render_status_sidebar()` — live component status, memory stats, navigation
- `render_memory_evaluation_page()` — search history view + RAGAS evaluation form

**Session state:** `SystemManager` is cached in `st.session_state` to persist across Streamlit rerenders.

**Path trick:** Inserts the project root into `sys.path` at startup so `from core.X import ...` works when launched from the `streamlit/` subdirectory.

**Connected to:** All `core/` modules, `utils/schemas.py`, `utils/config.py`

---

### `main.py`

**Role:** CLI demo/testing harness that exercises all HireFlow capabilities from the terminal.

**Class:** `HireFlowDemo`

Initialization order:
```
1. VectorStore().initialize()
2. HybridIndexer()
3. SearchRouter()
4. ReRanker(), ResumeParser(), JobParser()
5. RAGEvaluator(), MemoryRAG()
```

**Interactive menu options:**
1. Search Mode Comparison — runs vector, hybrid, and AI-enhanced search side by side
2. AI Evaluation Demo — parses a JD, evaluates top candidate
3. Quality Evaluation (RAGAS) — runs `RAGEvaluator.evaluate_search_quality()`
4. Memory System — shows `MemoryRAG` stats and recent searches
5. System Diagnostics — Pinecone stats, BM25 corpus sizes
6. Reload & Re-index Documents
7. Exit

**Connected to:** All `core/` modules, `utils/`

---

## Module Dependency Map

```
main.py / streamlit/app.py
    │
    ├── core/ingestion.py ──────────────► utils/utils.py (load_pdf)
    │
    ├── core/parsing.py ────────────────► utils/schemas.py
    │                                  ► utils/config.py
    │                                  ► utils/utils.py (clean_text, truncate_text)
    │
    ├── core/vector_store.py ───────────► utils/config.py
    │                                  ► utils/utils.py (get_embeddings)
    │
    ├── core/hybrid_indexer.py ─────────► core/vector_store.py
    │                                  ► utils/utils.py
    │
    ├── core/search_router.py ──────────► core/hybrid_indexer.py
    │                                  ► core/vector_store.py
    │                                  ► utils/config.py
    │                                  ► utils/utils.py
    │
    ├── core/re_ranker.py ──────────────► utils/schemas.py
    │                                  ► utils/config.py
    │                                  ► utils/utils.py
    │
    ├── core/memory_rag.py ─────────────► langchain-classic (ConversationBufferMemory)
    │
    ├── core/evaluator.py ──────────────► core/search_router.py
    │                                  ► ragas library
    │
    └── utils/multi_query.py ───────────► utils/config.py
                                       ► utils/utils.py
```

---

## Data Models

### Flow of a `Resume` through the system

```
PDF file
  ──ingestion──► Document(page_content=raw_text, metadata={candidate_id, name, ...})
  ──parsing───► Resume(candidate_id, name, skills, experience, text, ...)
  ──indexing──► Pinecone vector {id, values:[3072 floats], metadata:{...}}
              + BM25 corpus entry (lowercased text)
  ──search────► dict {candidate_id, name, combined_score, bm25_score, vector_score, text}
  ──ranking───► CandidateEvaluation(candidate_id, fit_score, strengths, gaps, risks, summary)
```

### `Document` (LangChain)
Used only during ingestion and indexing. Not persisted beyond `HybridIndexer`.

### `Resume` / `JobDescription` (Pydantic)
Structured data extracted by parsers. Used by `ReRanker` for evaluation.

### `CandidateEvaluation` (Pydantic)
Final output returned to the UI showing candidate fit assessment.

---

## Search Pipeline Deep-Dive

### Shallow Search (fast, vector-only)

```
query string
    ▼
GoogleGenerativeAIEmbeddings.embed_documents([query])  → 3072-dim float vector
    ▼
Pinecone.index.query(vector=..., top_k=K, filter={type:resume})
    ▼
List of {page_content, metadata, score}   ← score is cosine similarity [0,1]
```

### Deep Search (comprehensive, hybrid + LLM)

```
query string
    │
    ├─ BM25 path:
    │   query.lower().split() → tokens
    │   BM25Okapi.get_scores(tokens) → array of scores per document
    │   normalize by max score → [0,1]
    │
    └─ Vector path:
        embed query → Pinecone query → cosine scores [0,1]
    
    ▼
combine_results():
    bm25_results   = [{candidate_id, name, combined_score=bm25_normalized, ...}]
    vector_results = [{candidate_id, name, combined_score=cosine_score, ...}]
    all_results = bm25_results + vector_results   ← simple concatenation
    sort by combined_score descending
    return top_K

    ▼ (optional)
llm_rerank():
    Build prompt listing candidates with scores
    Ask Gemini: "Return reranked order as comma-separated numbers"
    Parse response → reorder list
```

### Scoring Notes
- BM25 and vector scores are both normalized to `[0,1]` before merging.
- There is **no fusion formula** — the two lists are literally appended and sorted. A BM25-only result and a vector-only result both have their score as `combined_score` with the other set to `0.0`.
- The `rrf_score` / `combined_rrf_score` field names in `search_router.py` are misleading — actual Reciprocal Rank Fusion is **not implemented**.

---

## Configuration & Environment

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hireflow          # optional, default: hireflow
GOOGLE_MODEL=gemini-2.5-flash         # optional, controls LLM model
```

### Data directories required at runtime:

```
data/
├── resumes/    ← place resume PDFs here
└── jds/        ← place job description PDFs here
```

These directories are not committed to git. You must create them before running.

---

## Key Design Decisions & Gotchas

| Topic | Detail |
|---|---|
| **VectorStore sharing** | `HybridIndexer` accepts an existing `VectorStore` to avoid two Pinecone connections. Always pass the app-level instance in production. |
| **Pinecone auto-create** | On first run `ensure_index()` creates the Pinecone index. If the dimension doesn't match it **deletes and recreates** the index, losing all stored vectors. |
| **BM25 is in-memory** | All BM25 data lives in RAM and is lost on restart. Resumes must be re-indexed on every startup. |
| **langchain-classic** | `MemoryRAG` imports `ConversationBufferMemory` from `langchain-classic`, a separate pinned package, **not** `langchain-community`. |
| **No prompts directory** | All LLM prompts are inline strings — there is no separate prompts folder despite architecture diagrams suggesting one. |
| **Quota handling** | `is_quota_error()` catches 429 / "quota exceeded" errors. On quota error, vector indexing/search is skipped gracefully and BM25 continues working. |
| **Text limits** | Parsers truncate input to 8000 chars. `MAX_TEXT_LENGTH` config (default 4000) is separate and used as a soft limit in other contexts. |
| **Embedding model** | `gemini-embedding-001` produces 3072-dim vectors. If you change this you must recreate the Pinecone index (dimension mismatch triggers auto-delete). |
| **Streamlit rerenders** | `SystemManager` is stored in `st.session_state` to survive Streamlit's rerender loop. Without this, all components would reinitialize on every interaction. |
| **MultiQueryGenerator** | Defined but **not** connected to any search path by default. Available for explicit use only. |
| **RAGAS evaluation** | `RAGEvaluator` creates a fresh `SearchRouter()` internally, independent of the app's main router instance. |
