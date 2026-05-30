# HireFlow — Code Explanation

A file-by-file walkthrough of every major code block in the project.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [utils/config.py](#utilsconfigpy)
3. [utils/schemas.py](#utilsschemaspy)
4. [utils/utils.py](#utilsutilspy)
5. [utils/multi_query.py](#utilsmulti_querypy)
6. [core/ingestion.py](#coreingestionpy)
7. [core/parsing.py](#coreparsingpy)
8. [core/vector_store.py](#corevector_storepy)
9. [core/hybrid_indexer.py](#corehybrid_indexerpy)
10. [core/search_router.py](#coresearch_routerpy)
11. [core/re_ranker.py](#corere_rankerpy)
12. [core/memory_rag.py](#corememory_ragpy)
13. [core/evaluator.py](#coreevaluatorpy)
14. [main.py](#mainpy)
15. [Data Flow Summary](#data-flow-summary)

---

## Project Structure

```
HireFlow/
├── core/                  # All AI/search business logic
│   ├── ingestion.py       # PDF → LangChain Document
│   ├── parsing.py         # LLM-powered structured extraction
│   ├── vector_store.py    # Pinecone vector DB wrapper
│   ├── hybrid_indexer.py  # BM25 + vector combined search
│   ├── search_router.py   # Smart query routing (shallow vs deep)
│   ├── re_ranker.py       # LLM candidate evaluation & scoring
│   ├── memory_rag.py      # Conversation memory tracking
│   └── evaluator.py       # RAGAS search quality metrics
├── utils/
│   ├── config.py          # Env-based configuration (pydantic-settings)
│   ├── schemas.py         # Pydantic data models
│   ├── utils.py           # Shared helper functions
│   └── multi_query.py     # Query expansion with LLM
├── streamlit/app.py       # Web UI
├── main.py                # CLI demo & interactive explorer
└── data/
    ├── resumes/           # Resume PDFs (not committed)
    └── jds/               # Job description PDFs (not committed)
```

---

## utils/config.py

**Purpose:** Centralised environment configuration using `pydantic-settings`.

```python
class AppConfig(BaseSettings):
    PINECONE_API_KEY: str = Field(default="", env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field(default="hireflow", env="PINECONE_INDEX_NAME")
    PINECONE_DIMENSION: int = Field(default=3072, env="PINECONE_DIMENSION")
    PINECONE_METRIC: str = Field(default="cosine", env="PINECONE_METRIC")
    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    LLM_MODEL: str = Field(default="gemini-2.5-flash", env="GOOGLE_MODEL")
    MAX_TEXT_LENGTH: int = Field(default=4000, env="MAX_TEXT_LENGTH")
```

- `BaseSettings` reads values from `.env` automatically (via `load_dotenv()`).
- `Field(env=...)` maps each attribute to its environment variable name.
- After the class, module-level constants (`GOOGLE_API_KEY`, `LLM_MODEL`, etc.) are exported so other files can do a flat `from utils.config import GOOGLE_API_KEY` without importing the whole config object.

---

## utils/schemas.py

**Purpose:** Pydantic data models that act as typed contracts between all components.

### `Resume`
| Field | Type | Description |
|---|---|---|
| `candidate_id` | str | Unique identifier (e.g. `c_john_doe`) |
| `name` | str | Candidate full name |
| `email` / `phone` / `location` | Optional | Contact details |
| `text` | str | Full resume raw text |
| `skills` | List[str] | Extracted skill tags |
| `experience` | Optional[int] | Total years of experience |

### `JobDescription`
| Field | Type | Description |
|---|---|---|
| `jd_id` | str | Unique identifier (e.g. `jd_accountant`) |
| `title` | str | Role title |
| `text` | str | Full JD raw text |
| `required_skills` | List[str] | Must-have skills |
| `optional_skills` | List[str] | Nice-to-have skills |
| `min_experience` | Optional[int] | Years of experience required |

### `CandidateEvaluation`
| Field | Type | Description |
|---|---|---|
| `candidate_id` | str | Links back to a `Resume` |
| `fit_score` | int (0–100) | Overall suitability score |
| `strengths` | List[str] | Key positives |
| `gaps` | List[str] | Missing requirements |
| `risks` | List[str] | Potential concerns |
| `summary` | str | Free-text evaluation |

---

## utils/utils.py

**Purpose:** Reusable helpers used across the entire codebase.

### `get_logger(name)`
```python
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    return logging.getLogger(name)
```
Creates a standardised logger with timestamps. Every module calls this at the top.

### `clean_text(text)`
```python
text = re.sub(r'\s+', ' ', text)
text = re.sub(r'[^\w\s\.\,\-\+\@\#\&\*\(\)]', '', text)
```
Collapses whitespace and strips unusual characters before sending text to the LLM.

### `truncate_text(text, max_length=8000)`
Prevents overlong prompts by cutting text and appending `"..."`.

### `load_pdf(file_path)`
```python
loader = PyPDFLoader(file_path)
documents = loader.load()
full_text = "\n".join([doc.page_content for doc in documents])
```
Uses LangChain's `PyPDFLoader` to extract multi-page PDF text, joins all pages into one string.

### `split_text(text, chunk_size, chunk_overlap)`
Uses `RecursiveCharacterTextSplitter` with a priority list of separators (`\n\n` → `\n` → ` ` → `""`). Returns a list of overlapping text chunks.

### `is_quota_error(error)`
```python
return "429" in str(error) or "quota" in error_str or "rate limit" in error_str
```
Detects Google/Pinecone API quota errors so callers can degrade gracefully instead of crashing.

### `get_embeddings()`
Creates a `GoogleGenerativeAIEmbeddings` instance using model `gemini-embedding-001` (768/3072-dim depending on config). Includes asyncio event loop fix for Streamlit thread compatibility.

### Filtering Helpers
| Function | What it does |
|---|---|
| `filter_by_skills(candidates, required_skills)` | Keeps candidates who possess ALL listed skills |
| `filter_by_location(candidates, target_locations)` | Case-insensitive partial-match on location |
| `filter_by_experience(candidates, min_experience)` | Keeps candidates meeting the year threshold |
| `apply_filters(candidates, ...)` | Chains all three filters sequentially |

---

## utils/multi_query.py

**Purpose:** Expands a single search query into multiple semantic variants to improve recall.

### `MultiQueryGenerator`

```python
def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
```
- **LLM path** (`generate_with_llm`): Sends the query to Gemini with a prompt asking for `num_queries` related but differently-worded queries. The response is split on newlines and cleaned.
- **Fallback path** (`generate_fallback`): Rule-based expansion — takes subsets of words from the original query (first 3 words, first + last word, single first word) to fill the quota without needing the LLM.

The result is a list of strings fed into hybrid/vector search to widen the net.

---

## core/ingestion.py

**Purpose:** Converts PDF files into LangChain `Document` objects with structured metadata.

### `load_resumes(directory)`
```python
doc = Document(
    page_content=text,
    metadata={
        'source': file_path,
        'filename': pdf_file,
        'candidate_id': f"c_{Path(pdf_file).stem}",
        'name': Path(pdf_file).stem.replace("_", " ").title()
    }
)
```
- Scans the given folder for `.pdf` files.
- Calls `load_pdf()` (from utils) on each.
- Wraps the text in a `Document` with automatic metadata: the filename stem becomes the candidate ID and display name (underscores → spaces, title-cased).

### `load_job_descriptions(directory)`
Same pattern as `load_resumes`, but uses `jd_id` and `title` as metadata keys.

### `DocumentProcessor` (legacy class)
Kept for backward compatibility. Thin wrappers around `load_pdf` and `split_text` with configurable `chunk_size` / `chunk_overlap`.

---

## core/parsing.py

**Purpose:** Uses Gemini LLM to extract _structured_ data from raw resume/JD text.

### `ResumeParser`

```python
chain = self.prompt_template | self.llm | self.output_parser
parsed_resume = chain.invoke({"resume_text": cleaned_text})
```

**How it works:**
1. **PromptTemplate** — injects the cleaned resume text and Pydantic format instructions.
2. **ChatGoogleGenerativeAI** — sends the prompt to Gemini at low temperature (0.1) for deterministic extraction.
3. **PydanticOutputParser** — validates the JSON response against the `Resume` Pydantic model.
4. Adds extra fields (`candidate_id`, `raw_text`, `parsing_method`) before returning a plain dict.

### `JobParser`
Identical pattern to `ResumeParser` but targets the `JobDescription` schema.

> **Key detail:** Text is pre-cleaned with `clean_text()` and capped at 8 000 characters with `truncate_text()` before being sent to the LLM to stay within context limits.

---

## core/vector_store.py

**Purpose:** Manages the Pinecone vector database — index lifecycle, embedding, and similarity search.

### `VectorStore.initialize()`
Bootstraps the full component stack in order:
1. Creates a `Pinecone` client with the API key.
2. Calls `ensure_index()` to create or repair the Pinecone index.
3. Calls `get_embeddings()` to get the Google embedding model.
4. Connects to the index and sets `_ready = True`.

### `ensure_index()`
```python
spec = ServerlessSpec(cloud="aws", region="us-east-1")
self.client.create_index(name=..., dimension=PINECONE_DIMENSION, metric=PINECONE_METRIC, spec=spec)
```
- If the index already exists with the correct dimension → no-op.
- If dimension **mismatches** → deletes and recreates (handles model upgrades).
- If missing → creates fresh. Waits 10 seconds after creation for Pinecone to become ready.

### `add_resumes(documents)` / `add_job_descriptions(documents)`
```python
vectors.append({
    "id": f"resume_{i}_{candidate_id}",
    "values": self.embeddings.embed_documents([doc.page_content])[0],
    "metadata": metadata   # includes page_content for retrieval
})
self.index.upsert(vectors=vectors)
```
Embeds each document's text and upserts it into Pinecone. The original `page_content` is stored inside the metadata so it can be returned alongside the score.

### `search_resumes(query, top_k, filters)`
```python
query_vector = self.embeddings.embed_documents([query])[0]
results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
```
Embeds the query, performs a cosine similarity search, and returns a list of `{page_content, metadata, score}` dicts.

### `convert_filters_to_pinecone(filters)`
Translates Python-style filters (`{"min": x, "max": y}`, lists, scalars) into Pinecone's `$gte/$lte/$eq/$in` filter syntax.

---

## core/hybrid_indexer.py

**Purpose:** Combines BM25 keyword search with vector semantic search for better recall and precision.

### Why Hybrid?
- **BM25** (lexical) is good at exact keyword matches (e.g. "QuickBooks", "CPA").
- **Vector** (semantic) is good at conceptual matches (e.g. "financial professional" ≈ "accountant").
- Combining them covers both exact and fuzzy matches.

### `index_resumes(resumes)`
```python
# BM25 side
self.resume_texts.append(text.lower())
self.resume_metadata.append(resume.metadata)
tokenized_texts = [text.split() for text in self.resume_texts]
self.bm25_resumes = BM25Okapi(tokenized_texts)

# Vector side
self.vector_store.add_resumes(resumes)
```
- Appends to the existing corpus (accumulative, not replacing) so multiple upload calls work correctly.
- Rebuilds the `BM25Okapi` model over the full accumulated corpus each time.
- Simultaneously sends documents to Pinecone.

### `search_resumes(query, top_k)`
```python
bm25_scores = self.bm25_resumes.get_scores(query_tokens)
vector_results = self.vector_store.search_resumes(query, top_k * 2, filters={"type": "resume"})
combined = self.combine_results(bm25_scores, vector_results, top_k, is_jd=False)
```
Runs both searches in parallel (sequentially in code) and delegates merging to `combine_results`.

### `combine_results(bm25_scores, vector_results, top_k, is_jd)`
```
BM25 scores normalised to [0, 1] by dividing by max BM25 score
Vector scores are already in [0, 1] (cosine similarity)
Both lists concatenated → sorted by combined_score descending → top_k returned
```
> **Note:** This is a simple concatenate-and-sort, **not** true Reciprocal Rank Fusion (despite `rrf_score` field names in earlier versions). Each candidate carries `bm25_score`, `vector_score`, and `combined_score`.

---

## core/search_router.py

**Purpose:** Intelligently routes queries to either a fast (shallow) or thorough (deep) search strategy.

### The Two Strategies

| Strategy | Method | Use Case |
|---|---|---|
| **Shallow** | Vector-only (`VectorStore.search_resumes`) | Short, specific queries ("Python developer") |
| **Deep** | Hybrid BM25+Vector + LLM rerank (`HybridIndexer.search_resumes`) | Long, nuanced queries |

### `build_search_chain()`
Builds a LangChain `RunnableBranch`:
```python
chain = RunnableBranch(
    (lambda x: route_search(x) == "shallow", shallow_search),
    (lambda x: route_search(x) == "deep",    deep_search),
    deep_search  # default fallback
)
```

### `route_search(inputs)` — the routing logic
```
1. If search_mode == "shallow" or "deep" → use that directly
2. Else if LLM available → ask Gemini: "shallow or deep for this query?"
3. Else fallback heuristic → word count ≤ 3 → shallow, else → deep
```

### `llm_rerank(results, query, jd_context, top_k)`
Sends the current ranked list to Gemini and asks it to return the numbers in a new order based on semantic fit. Parses the `"3,1,2,4,5"` style response and reorders results accordingly.

---

## core/re_ranker.py

**Purpose:** LLM-powered deep evaluation of a single candidate against a job description.

### `evaluate_candidate(resume, jd)`
```python
prompt = f"""
Analyze this candidate for the job:
JOB: {jd.text}
CANDIDATE: {resume.text}
Give me: 1. 3 key strengths  2. 3 areas for improvement  3. Any risks  4. A brief summary
"""
response = self.llm.invoke(prompt)
```
Parses the LLM's free-text response into structured lists using `extract_section()` and `extract_summary()`.

**Fit score formula:**
```
fit_score = 50 + (num_strengths × 15) − (num_gaps × 10)
clamped to [0, 100]
```

### `simple_evaluation(resume, jd)` — rule-based fallback
When the LLM is unavailable, checks:
- Experience: meets `min_experience` → +20 pts
- Skills: at least one required skill present → +20 pts

### `re_rank_candidates(candidates, jd)`
Iterates over all candidates, calls `evaluate_candidate` for each, then sorts the resulting `CandidateEvaluation` list by `fit_score` descending.

### `extract_section(text, section_name, max_items)`
Line-by-line parser that finds the matching section header, then reads bullet points (`-`, `•`, `*`, numbered lists) until the next section header or end of text.

---

## core/memory_rag.py

**Purpose:** Maintains a session-scoped conversation history of searches and candidate views.

```python
self.memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

Uses LangChain's `ConversationBufferMemory` (from `langchain-classic`) as an in-memory store.

### Key Methods

| Method | What it stores |
|---|---|
| `record_search(query, results_count)` | `"Search: <query>"` as a `HumanMessage`, `"Found N results"` as `AIMessage` |
| `record_candidate_view(candidate_name)` | `"Viewed candidate: <name>"` |
| `get_search_history()` | Returns last 5 queries by filtering `HumanMessage` objects starting with `"Search:"` |
| `get_memory_stats()` | Returns `{total_messages, search_count, candidate_views}` |

> Memory is **in-session only** — it resets when the app restarts.

---

## core/evaluator.py

**Purpose:** Measures the quality of the search pipeline using RAGAS (Retrieval-Augmented Generation Assessment).

### `RAGEvaluationMetrics` (dataclass)
```python
answer_relevancy: float    # Is the answer relevant to the question?
context_precision: float   # How precise is the retrieved context?
faithfulness: float        # Is the answer faithful to the context?
answer_correctness: float  # Is the answer factually correct?
overall_score: float       # Weighted average
```

### Evaluation Pipeline

```
evaluate_search_quality(query, expected_skills)
  └─> search_router.search(query)         # Run the actual search
  └─> evaluate_with_ragas(candidates)     # Score with RAGAS
        └─> prepare_ragas_data()          # Build a DataFrame with columns:
            │   question | contexts | ground_truth | answer
        └─> ragas.evaluate(df, metrics)   # RAGAS scoring
```

### `prepare_ragas_data(query, candidates, expected_skills)`
For each candidate:
- **context**: `"Name: X | Skills: a, b, c | Experience: N years"`
- **ground_truth**: generated by `create_ground_truth()` — `"Excellent match: 4/5 skills"` etc.
- **answer**: `"<Name> with skills in a, b, c"`

### `calculate_overall_score(metrics)`
```
overall = 0.30 × answer_relevancy
        + 0.30 × context_precision
        + 0.20 × faithfulness
        + 0.20 × answer_correctness
```

### History
Every evaluation is appended to `self.evaluation_history`. `get_evaluation_summary()` returns averages and the last 5 records. `export_evaluations(filename)` dumps history to CSV.

---

## main.py

**Purpose:** CLI entry point and interactive demo runner that exercises every component.

### `HireFlowDemo` class

| Method | What it does |
|---|---|
| `check_prerequisites()` | Validates API keys and that `data/resumes/` and `data/jds/` contain PDFs |
| `initialize_system()` | Instantiates all 8 components (`VectorStore`, `HybridIndexer`, `SearchRouter`, `ReRanker`, `ResumeParser`, `JobParser`, `RAGEvaluator`, `MemoryRAG`) |
| `load_and_index_documents()` | Loads PDFs → indexes in hybrid indexer and vector store |
| `test_search_modes()` | Runs the same query through vector-only, hybrid, and router+rerank; prints results side-by-side |
| `test_ai_evaluation()` | Parses a JD and runs `ReRanker.evaluate_candidate_fit` on the top result |
| `test_quality_metrics()` | Runs `RAGEvaluator.evaluate_search_quality` and prints RAGAS scores |
| `test_memory_system()` | Shows stats and recent search history from `MemoryRAG` |
| `interactive_menu()` | REPL loop with 7 menu options for manual exploration |

### Startup flow
```
main()
  └─> show_banner()
  └─> check_prerequisites()   → exit if fail
  └─> initialize_system()     → exit if fail
  └─> load_and_index_documents() → exit if fail
  └─> test_search_modes()
  └─> test_ai_evaluation()
  └─> test_memory_system()
  └─> check_system_status()
  └─> interactive_menu()      → loops until user exits
```

---

## Data Flow Summary

```
PDF files
   │
   ▼
ingestion.py          load_resumes / load_job_descriptions
   │   (LangChain Document + metadata)
   ▼
parsing.py            ResumeParser / JobParser
   │   (Gemini LLM → Resume / JobDescription Pydantic objects)
   ▼
hybrid_indexer.py     index_resumes / index_job_descriptions
   ├── BM25Okapi       (in-memory keyword index)
   └── VectorStore     (Pinecone vector index via Google embeddings)

──────────── SEARCH TIME ────────────

User query
   │
   ▼
multi_query.py        MultiQueryGenerator (optional query expansion)
   │
   ▼
search_router.py      SearchRouter.search()
   │   RunnableBranch routes to:
   ├── shallow → VectorStore.search_resumes()
   └── deep    → HybridIndexer.search_resumes()
                    └── LLM rerank via llm_rerank()
   │
   ▼
re_ranker.py          ReRanker.evaluate_candidate() per result
   │   (Gemini LLM → CandidateEvaluation with fit_score)
   │
   ▼
memory_rag.py         MemoryRAG.record_search()
   │
   ▼
evaluator.py          RAGEvaluator.evaluate_search_quality()
   │   (RAGAS metrics on final results)
   ▼
Final ranked candidates with fit scores
```

---

> Generated by Claude Code · HireFlow · May 2026
