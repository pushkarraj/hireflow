# HireFlow Concepts Documentation — Design Spec
**Date:** 2026-05-29  
**Status:** Approved  

---

## Goal

Create a single comprehensive guide (`CONCEPTS.md`) at the project root that teaches beginners to GenAI how HireFlow works — covering both the underlying AI/ML concepts and the actual code implementation — using a story-driven, end-to-end narrative structure.

## Audience

Students who know Python but have no prior experience with embeddings, vector databases, LLMs, RAG pipelines, or semantic search. Concepts must be explained from scratch using real-world analogies before introducing technical terms.

## Document: `CONCEPTS.md`

Placed at the project root alongside `README.md`. Single file, navigable via a Table of Contents with anchor links.

---

## Structure & Content

### Introduction: The Problem We're Solving
- Why keyword/grep search fails at scale for recruiting (semantic mismatch example)
- What semantic search means in plain English
- Full pipeline diagram (text-based ASCII or Mermaid) showing all 8 stages
- How to read this guide (each chapter = one stage of the pipeline)

---

### Chapter 1 — Getting Documents Into the System (`ingestion.py`)
**Concept:** Document pipelines — why raw PDFs can't go directly into AI models  
**Topics:**
- PDF text extraction with PyPDF
- LangChain `Document` objects as standardized data containers
- Metadata: why attaching `candidate_id`, `name`, `skills` to text matters for filtering later
- Text chunking: what happens when a resume exceeds an LLM's context window  

**Code walkthrough:** `load_resumes()`, `DocumentProcessor`, metadata attachment  
**Design decisions:** Why LangChain Documents instead of plain dicts  
**What can go wrong:** Scanned PDFs (no extractable text), encoding errors, malformed PDFs  

---

### Chapter 2 — Making Machines Read Resumes (`parsing.py`)
**Concept:** Structured vs unstructured data; prompt engineering for extraction  
**Topics:**
- What "unstructured text" means and why machines can't use it directly
- Pydantic schemas as data contracts (`Resume`, `JobDescription`)
- How LLMs extract structured fields via prompting (with example prompt → response)
- Why validation matters (wrong types, missing required fields)  

**Code walkthrough:** `ResumeParser.parse()`, `JobParser.parse_job_description()`, LLM invocation pattern  
**Design decisions:** LLM extraction vs regex; why Pydantic over dataclasses  
**What can go wrong:** LLM hallucination on structured extraction, missing emails, inconsistent skill formatting  

---

### Chapter 3 — Teaching Computers to Understand Meaning (`vector_store.py`)
**Concept:** Embeddings, vector space, cosine similarity, vector databases  
**Topics:**
- What an embedding is ("a fingerprint for meaning" analogy)
- Embedding space: why similar sentences end up near each other mathematically
- Cosine similarity vs Euclidean distance — which measures "direction of meaning"
- Why you can't just `numpy.argmax` over millions of vectors (scale problem)
- Pinecone as a managed ANN (Approximate Nearest Neighbor) index
- Google GenAI embeddings: 768 dimensions, `models/gemini-embedding-001`  

**Code walkthrough:** `VectorStore.initialize()`, `ensure_index()` (ServerlessSpec, dimension checking), `add_resumes()` (embed → upsert), `search_resumes()` (embed query → Pinecone query → return matches), `convert_filters_to_pinecone()`  
**Design decisions:** Why cosine over dot product; why 768 dims; why AWS us-east-1 ServerlessSpec; storing `page_content` in metadata  
**What can go wrong:** Dimension mismatch on existing index (code auto-recreates), API quota errors (handled via `is_quota_error`), cold-start latency on first upsert  

---

### Chapter 4 — The Best of Both Worlds: Hybrid Search (`hybrid_indexer.py`)
**Concept:** BM25 keyword search + vector semantic search; why neither alone is enough  
**Topics:**
- BM25 from first principles: term frequency, inverse document frequency, Okapi formula (plain English)
- Where pure vector search fails: exact terms like "CPA", "Series B", specific company names
- Where pure BM25 fails: synonyms, paraphrasing, semantic intent
- Score fusion: combining two ranked lists into one
- What true Reciprocal Rank Fusion (RRF) looks like vs what this code does (score concatenation)  

**Code walkthrough:** `HybridIndexer.index_resumes()` (BM25Okapi construction), `search_resumes()` (parallel BM25 + vector query), `combine_results()` (merge, sort by combined_score)  
**Design decisions:** Why in-memory BM25 (simplicity for teaching); acknowledged gap: true RRF would give better fusion  
**What can go wrong:** BM25 index lost on restart (in-memory), mismatched result counts between BM25 and vector, empty results if index not built before searching  

---

### Chapter 5 — Smart Query Routing (`search_router.py`)
**Concept:** LangChain RunnableBranch; cost/latency tradeoffs in search systems  
**Topics:**
- Why not always run deep search? (Gemini API cost, latency per query)
- The concept of a "router" in AI pipelines
- LangChain `RunnableBranch` as a composable decision tree
- LLM-based query classification with word-count fallback
- Shallow (vector-only, fast) vs Deep (hybrid + LLM rerank, thorough)  

**Code walkthrough:** `SearchRouter.build_search_chain()`, `route_search()` (LLM routing logic + heuristic fallback), `shallow_search()`, `deep_search()`, `llm_rerank()`, `search()` entry point  
**Design decisions:** Why auto-routing instead of always deep; why the word-count heuristic as fallback  
**What can go wrong:** LLM router misclassification, deep search fallback on shallow failure, `RunnableBranch` evaluating route function twice  

---

### Chapter 6 — AI as the Final Judge (`re_ranker.py`)
**Concept:** Retrieval vs ranking; LLM-based candidate evaluation  
**Topics:**
- The difference between finding candidates (retrieval) and ranking them (evaluation)
- Why a similarity score of 0.87 doesn't mean "good fit"
- What a fit score, strengths, gaps, and risks mean in recruiting context
- How the `CandidateEvaluation` Pydantic model enforces output structure  

**Code walkthrough:** `ReRanker.evaluate_candidate_fit()` — prompt construction, LLM call, response parsing into `CandidateEvaluation`  
**Design decisions:** Why re-ranking happens after retrieval (not during); hardcoded prompts vs prompt templates  
**What can go wrong:** LLM returns malformed JSON, fit score outside 0–100, empty strengths/gaps lists  

---

### Chapter 7 — Did We Get It Right? (`evaluator.py`)
**Concept:** RAG evaluation; RAGAS metrics  
**Topics:**
- Why you need automated quality measurement (can't manually review 1000 queries)
- The four RAGAS metrics in plain English:
  - **Answer Relevancy:** does the answer match what was asked?
  - **Context Precision:** are the retrieved documents actually useful?
  - **Faithfulness:** does the answer stick to what the documents say?
  - **Answer Correctness:** is the answer factually right?
- What an "overall score" hides  

**Code walkthrough:** `RAGEvaluator.evaluate_search_quality()` — search execution, metric calculation, `RAGASMetrics` result object  
**Design decisions:** Why RAGAS over manual evaluation; limitations of automated metrics  
**What can go wrong:** Metric gaming (high scores with bad results), RAGAS dependency version sensitivity, slow evaluation on large result sets  

---

### Chapter 8 — Remembering What Was Asked (`memory_rag.py`)
**Concept:** Stateless vs stateful AI; conversation memory patterns  
**Topics:**
- Why stateless search loses context between queries
- LangChain `ConversationBufferMemory` as a session log
- `HumanMessage` / `AIMessage` as the storage format
- What gets recorded: queries, result counts, candidate views  

**Code walkthrough:** `MemoryRAG.record_search()`, `get_search_history()`, `get_memory_stats()`  
**Design decisions:** Why `langchain-classic` for memory (not `langchain-community`); buffer vs summary memory  
**Honest limitation:** In-session only — contrast with persistent memory (Redis, database-backed) and what that would require  

---

### Chapter 9 — The Glue: Utils & Config (`utils/`)
**Concept:** Configuration management; data contracts across layers; query expansion  
**Topics:**
- `config.py`: pydantic-settings for type-safe env vars; flat module-level constants pattern
- `schemas.py`: why every layer speaks the same language (`Resume`, `JobDescription`, `CandidateEvaluation`)
- `multi_query.py`: query expansion — why "senior accountant" → 3 variants improves recall
- `utils.py`: `get_embeddings()`, `get_logger()`, `load_pdf()`, `is_quota_error()`, `clean_text()`, `split_text()` — the shared toolkit; also post-search filter helpers: `filter_by_skills()`, `filter_by_location()`, `filter_by_experience()`, `apply_filters()`  

---

### Closing: The Full Journey — One Resume, One Query, End to End
A single concrete worked example tracing:
1. `Charles_Clark_Resume_47.pdf` → ingested → parsed → embedded → indexed (Chapters 1–3)
2. Query: `"senior accountant with QuickBooks and tax experience"` → routed → hybrid searched → reranked → evaluated (Chapters 4–7)

Shows actual data shape at each transformation step. Ends with a "What to build next" list of improvements students can make: persistent memory, true RRF, streaming Streamlit UI, multi-JD batch search.

---

## File Location

`CONCEPTS.md` at the project root (`/`), alongside `README.md`.

## Length Estimate

~4,000–6,000 words. Long enough to be thorough, short enough to read in one sitting.

## Non-Goals

- Does not replace inline code comments
- Does not cover Streamlit UI internals in depth (mentioned in intro only)
- Does not require students to run the code to follow along (all snippets are self-contained)
