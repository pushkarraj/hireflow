# main.py — Simple Line-by-Line Breakdown

---

## Lines 1–14 | File Description (Docstring)

```python
"""
HireFlow - Comprehensive AI-Powered Candidate Search Engine
...
"""
```

This is just a big comment at the top of the file. It tells anyone reading the code what the file does — it's the **starting point** of the whole HireFlow app. Think of it like a sticky note on the front of a notebook.

---

## Lines 16–18 | Fix SSL Certificate Issues

```python
import truststore
truststore.inject_into_ssl()
```

Some companies use a security tool (like Zscaler) that sits between your computer and the internet. This can break secure connections (HTTPS). These two lines tell Python to **trust your Mac's built-in certificates** so the app can connect to the internet without errors.

---

## Lines 20–22 | Import Standard Python Tools

```python
import os
import sys
from typing import Dict, List
```

- `os` — lets Python talk to your operating system (e.g., check if a folder exists)
- `sys` — lets Python interact with the terminal (e.g., exit with a code)
- `Dict, List` — just type hints so we can say "this variable is a list" or "this is a dictionary"

---

## Lines 23–30 | Import HireFlow Core Components

```python
from core.ingestion import load_resumes, load_job_descriptions
from core.vector_store import VectorStore
from core.hybrid_indexer import HybridIndexer
from core.search_router import SearchRouter
from core.parsing import ResumeParser, JobParser
from core.re_ranker import ReRanker
from core.evaluator import RAGEvaluator
from core.memory_rag import MemoryRAG
```

These lines bring in all the main building blocks of HireFlow:

| Import | What it does (simply) |
|--------|----------------------|
| `load_resumes` / `load_job_descriptions` | Reads PDF files from a folder |
| `VectorStore` | Connects to Pinecone to store/search resume embeddings |
| `HybridIndexer` | Combines two search methods: keyword (BM25) + semantic (vector) |
| `SearchRouter` | Decides which search method to use based on the query |
| `ResumeParser` / `JobParser` | Uses AI to extract structured info from PDF text |
| `ReRanker` | Uses AI to score how well a candidate fits a job |
| `RAGEvaluator` | Measures how good the search results are |
| `MemoryRAG` | Remembers what searches were done in this session |

---

## Lines 31–33 | Import Utilities

```python
from utils.config import GOOGLE_API_KEY, PINECONE_API_KEY
from utils.utils import get_logger, load_pdf
from utils.schemas import JobDescription
```

- `GOOGLE_API_KEY`, `PINECONE_API_KEY` — the secret keys needed to use Google AI and Pinecone
- `get_logger` — sets up a log printer so we can track what's happening
- `load_pdf` — reads text out of a PDF file
- `JobDescription` — a structured template (Pydantic model) for a job description

---

## Line 35 | Set Up Logger

```python
logger = get_logger(__name__)
```

Creates a logger for this file. It's like a diary — when something important happens (or goes wrong), it writes a note about it.

---

## Lines 37–54 | Class Definition + `__init__`

```python
class HireFlowDemo:
    def __init__(self):
        self.components = { 'vector_store': None, ... }
        self.system_ready = False
        self.resume_dir = "data/resumes"
        self.jd_dir = "data/jds"
```

This creates the main **HireFlow Demo class**. When you create an instance of it:
- It makes a dictionary called `components` to hold all the AI tools (all set to `None` until initialized)
- `system_ready` is `False` — nothing is running yet
- It remembers where the resume PDFs and job description PDFs are stored

---

## Lines 56–61 | `show_banner()`

```python
def show_banner(self):
    print("="*70)
    print("HireFlow - AI-Powered Candidate Search Engine")
    ...
```

Just prints a nice welcome header in the terminal when you run the app. Purely cosmetic.

---

## Lines 63–95 | `check_prerequisites()`

```python
def check_prerequisites(self) -> bool:
```

Before the app starts, this method checks that everything needed is in place:

1. **Is the Google API key set?** — Without this, the AI (Gemini) won't work
2. **Is the Pinecone API key set?** — Without this, the vector database won't work
3. **Do the data folders exist?** (`data/resumes/` and `data/jds/`)
4. **Are there any PDF files in those folders?**

If any check fails, it prints an error and returns `False` (meaning "stop, something's missing").

---

## Lines 97–142 | `initialize_system()`

```python
def initialize_system(self) -> bool:
```

This boots up all the AI components one by one. Think of it like turning on different machines in a factory:

1. **VectorStore** — Connects to Pinecone (the cloud vector database)
2. **HybridIndexer** — Prepares the combined BM25 + vector search engine
3. **SearchRouter** — Sets up the "traffic controller" that picks the right search method
4. **ReRanker, ResumeParser, JobParser** — Fires up the Gemini AI models
5. **RAGEvaluator** — Prepares the search quality tester
6. **MemoryRAG** — Starts the session memory tracker

If anything fails, it returns `False`. If everything works, `system_ready` is set to `True`.

---

## Lines 144–179 | `load_and_index_documents()`

```python
def load_and_index_documents(self) -> bool:
```

This loads all the PDF files and stores them in the search index:

1. **Load resumes** — reads all resume PDFs from `data/resumes/`
2. **Load job descriptions** — reads all JD PDFs from `data/jds/`
3. **Clear Pinecone** — wipes the cloud database so we start fresh
4. **Index resumes** — converts resume text to vectors and stores them (in both BM25 and Pinecone)
5. **Index job descriptions** — does the same for job descriptions

After this step, the system is ready to search.

---

## Lines 181–223 | `test_search_modes()`

```python
def test_search_modes(self):
```

Demonstrates three different ways to search for candidates using the same query:
`"senior accountant with Excel and QuickBooks experience"`

| Mode | How it works |
|------|-------------|
| **Vector Search** | Finds resumes that *mean* the same thing (semantic similarity) |
| **Hybrid Search** | Combines keyword matching (BM25) + semantic matching |
| **AI-Enhanced Search** | Hybrid search + AI re-ranking to pick the best matches |

Results are printed and the query is saved to memory.

---

## Lines 225–268 | `test_ai_evaluation()`

```python
def test_ai_evaluation(self):
```

Tests the AI's ability to judge how well a candidate fits a job:

1. Loads a sample job description PDF
2. Searches for `"senior accountant"` candidates
3. Parses the job description into a structured format
4. Sends the top candidate + job description to Gemini AI
5. Prints the **fit score** (0–100), **strengths**, **gaps**, and a **summary**

---

## Lines 270–302 | `test_quality_metrics()`

```python
def test_quality_metrics(self):
```

Measures how *good* the search results actually are using **RAGAS** (a standard AI evaluation framework):

| Metric | What it measures |
|--------|----------------|
| Answer Relevancy | Are results related to the query? |
| Context Precision | Are the top results actually the best ones? |
| Faithfulness | Does the AI's response match the source data? |
| Answer Correctness | Are the expected skills present in the results? |
| Overall Score | Weighted average of all the above |

Grades: **≥0.8** = Excellent, **≥0.6** = Good, **<0.6** = Needs improvement.

---

## Lines 304–323 | `test_memory_system()`

```python
def test_memory_system(self):
```

Shows what the in-session memory has recorded so far:
- How many messages are stored
- How many searches were made
- How many candidate profiles were viewed
- The last 3 search queries

This memory only lasts for the current session (it resets when you close the app).

---

## Lines 325–347 | `check_system_status()`

```python
def check_system_status(self):
```

Prints a health report of the running system:
- **VectorStore**: Is it connected? How many vectors are stored? What's the dimension size?
- **HybridIndexer**: How many resumes and JDs are indexed? Are all systems ready?

---

## Lines 349–360 | `display_results()`

```python
def display_results(self, results: List[Dict], max_show: int = 3):
```

A helper function that neatly prints candidate search results. For each result it shows:
- Candidate name
- Match score
- Years of experience
- Top 3 skills

It handles both vector search results and deep/hybrid search results (which store data slightly differently).

---

## Lines 362–405 | `interactive_menu()`

```python
def interactive_menu(self):
```

This is an interactive terminal menu that loops forever until you exit. It shows 7 options:

```
1. Search Mode Comparison
2. AI Evaluation Demo
3. Quality Evaluation (RAGAS)
4. Memory System
5. System Diagnostics
6. Reload & Re-index Documents
7. Exit
```

You type a number, it runs the corresponding function. If you press `Ctrl+C`, it exits cleanly.

---

## Lines 407–450 | `main()`

```python
def main():
```

This is the **master controller** that runs everything in order when you type `python main.py`:

```
Step 1: Show the banner
Step 2: Check prerequisites  → stop if anything is missing
Step 3: Initialize all components → stop if any component fails
Step 4: Load and index all PDFs → stop if loading fails
Step 5: Auto-run all 4 demos (search, evaluation, memory, diagnostics)
Step 6: Launch the interactive menu
```

Returns `0` if everything went well, `1` if there was an error.

---

## Lines 452–453 | Entry Point

```python
if __name__ == "__main__":
    sys.exit(main())
```

This is the standard Python way to say:
> "Only run `main()` if this file is being run directly (not imported by another file)."

`sys.exit(main())` passes the return code (`0` = success, `1` = error) back to the terminal.

---

*Generated for HireFlow — Resume to Job Search Engine*
