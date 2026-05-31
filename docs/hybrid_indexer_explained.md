# HybridIndexer — Explained in Plain English

> **File:** `core/hybrid_indexer.py`
> **What it is:** The core search engine that finds matching candidates for a job by combining two types of search simultaneously.

---

## The Big Picture

Imagine you're running a **recruitment agency** with a filing cabinet full of resumes. When a company gives you a job description, your job is to find the best matching candidates fast.

The `HybridIndexer` is your **super-smart filing assistant** that does two types of searches at the same time — and then combines the results.

---

## The Two Types of Search

### 1. Keyword Search — BM25

Think of this like a **Ctrl+F search** on steroids.

> *"I need someone with QuickBooks and Excel"*

BM25 goes through every resume and literally counts how many times those exact words appear. The resume with the most matches gets the highest score.

**Analogy:** You're hiring a librarian. You search your shelf for books containing the words "library" and "cataloguing." The book with those words on every page ranks #1.

**Limitation:** If a resume says "financial software" but the job says "QuickBooks" — BM25 misses it. It only matches exact words.

---

### 2. Meaning-Based Search — Vector / Pinecone

This one is much smarter. It doesn't look for exact words — it understands **meaning**.

> *"QuickBooks"* and *"accounting software"* mean the same thing → this search knows that.

Every resume is converted into a list of numbers (called an **embedding** or **vector**) that captures the *meaning* of the text. Similar meanings produce similar numbers, which means a high match score.

**Analogy:** You ask an experienced recruiter: *"Find someone good with accounting tools."* They don't just search for the word "QuickBooks" — they think: *"People who know Excel, SAP, Tally, or FreshBooks are also relevant."*

The vectors are stored in **Pinecone** (a cloud database built exactly for this), managed by `core/vector_store.py`.

---

## Step 1 — Storing Resumes (Indexing)

Before you can search, you have to **file away** all the resumes. This is called *indexing*.

```
PDF Resume → extract text → store two copies:
   1. Plain text list      → for BM25 keyword search  (stays in memory)
   2. Convert to numbers   → send to Pinecone cloud    (vector search)
```

**In code** — `index_resumes()` at line 40:

```python
for resume in resumes:
    text = resume.page_content.lower()           # Get the plain text
    self.resume_texts.append(text)               # Store for BM25
    self.resume_metadata.append(resume.metadata) # Store name, ID, etc.

# Build the keyword search engine over ALL stored texts
self.bm25_resumes = BM25Okapi(tokenized_texts)

# Also send to Pinecone for meaning-based search
self.vector_store.add_resumes(resumes)
```

**Analogy:** You photocopy every resume. One copy goes into a keyword-indexed binder (BM25). The other gets summarised into a number-code and stored in a cloud database (Pinecone).

---

## Step 2 — Searching

When a recruiter types *"senior accountant with Excel and QuickBooks experience"*, here's what happens inside `search_resumes()` at line 119:

### Round 1 — BM25 Keyword Search

```python
query_tokens = query.lower().split()
# → ["senior", "accountant", "excel", "quickbooks"]

bm25_scores = self.bm25_resumes.get_scores(query_tokens)
# Scores every resume by exact word frequency
```

Example output:

| Resume | BM25 Score |
|---|---|
| John Smith (mentions QuickBooks 3×) | 8.4 |
| Jane Doe (mentions Excel 5×) | 7.1 |
| Bob Lee (no mention) | 0.2 |

### Round 2 — Vector Semantic Search

```python
vector_results = self.vector_store.search_resumes(query, top_k * 2)
# Pinecone finds resumes whose *meaning* is closest to the query
```

Example output:

| Resume | Vector Score |
|---|---|
| Jane Doe ("financial software tools") | 0.89 |
| Alice Wong ("bookkeeping systems") | 0.85 |
| John Smith | 0.72 |

---

## Step 3 — Combining Results

Two separate ranked lists now exist. `combine_results()` at line 177 merges them.

**First, BM25 scores are normalised to [0–1]** so they're on the same scale as vector scores:

```python
max_bm25 = max(bm25_scores)            # e.g. 8.4
normalized = score / max_bm25          # John Smith: 8.4/8.4 = 1.0
```

**Then both lists are concatenated and sorted:**

```python
all_results = bm25_results + vector_results
all_results.sort(key=lambda x: x['combined_score'], reverse=True)
return all_results[:top_k]
```

Final merged ranking:

| Resume | Score | Source |
|---|---|---|
| John Smith | 1.00 | BM25 |
| Jane Doe | 0.89 | Vector |
| Alice Wong | 0.85 | Vector |

> **Note:** The field name `rrf_score` in the code suggests "Reciprocal Rank Fusion," but the actual implementation is a simpler concatenate-and-sort. It still works well — just not mathematically optimal RRF.

---

## Full Flow, Start to Finish

```
YOU TYPE: "senior accountant with Excel experience"
              │
              ▼
    ┌─────────────────────┐
    │   BM25 Search       │  → Finds exact keyword matches
    └─────────────────────┘
              │    (runs at the same time)
    ┌─────────────────────┐
    │   Pinecone Search   │  → Finds meaning-based matches
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  combine_results()  │  → Normalise + merge + sort
    └─────────────────────┘
              │
              ▼
       TOP N CANDIDATES
```

---

## Key Classes and Functions

| Name | What It Does | Location |
|---|---|---|
| `HybridIndexer` | Orchestrates both search engines | `core/hybrid_indexer.py` L14 |
| `BM25Okapi` | Keyword search engine (external library) | used at L66 |
| `VectorStore` | Talks to Pinecone for semantic search | `core/vector_store.py` L20 |
| `index_resumes()` | Files resumes into both engines | L40 |
| `index_job_descriptions()` | Same but for job descriptions | L83 |
| `search_resumes()` | Runs both searches on a query | L119 |
| `combine_results()` | Merges the two result lists | L177 |
| `get_index_stats()` | Reports readiness of both engines | L270 |
| `SearchRouter` | Decides when to use HybridIndexer (deep mode only) | `core/search_router.py` L17 |

---

## One-Sentence Summary

> **HybridIndexer stores resumes in two ways — a keyword index and a meaning-based cloud database — then when you search, it runs both simultaneously and merges the results into one ranked list, giving you the best of exact-word matching AND intelligent meaning-matching.**
