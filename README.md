# HireFlow - Intelligent Candidate Search and Evaluation

An AI-powered recruitment platform that combines semantic search with Generative AI to intelligently match candidates to job descriptions. HireFlow goes beyond traditional keyword-based filtering to understand the deeper semantic alignment between candidates and roles.

## 🚀 Features

- **Semantic Document Processing**: Parse resumes and job descriptions using LangChain-powered LLM parsing
- **Intelligent Vector Search**: Use Pinecone vector database for semantic similarity matching
- **AI-Powered Evaluation**: Generate candidate evaluations with confidence scores and insights
- **Modern Web Interface**: Streamlit-based recruiter dashboard
- **Modular Architecture**: Clean, maintainable codebase with unified logging

## 🏗️ Architecture

```
HireFlow/
├── core/                    # Core pipeline components
│   ├── ingestion.py        # Document loading and text splitting
│   ├── parsing.py          # LLM-based resume/JD parsing
│   ├── indexing.py         # Vector indexing with Pinecone
│   ├── retrieval.py        # Semantic search and retrieval
│   ├── filters.py          # Candidate filtering logic
│   ├── evaluation.py       # Candidate evaluation
│   └── re_ranker.py        # Result re-ranking
├── utils/                   # Shared utilities
│   ├── config.py           # Configuration management
│   ├── schemas.py          # Pydantic data models
│   ├── embeddings.py       # Embedding client
│   └── utils.py            # General utilities
├── prompts/                 # LLM prompt templates
├── streamlit/               # Web interface
│   └── app.py              # Main Streamlit application
├── data/                    # Sample data
│   ├── resumes/            # Resume PDFs
│   └── jds/                # Job description PDFs
└── main.py                  # CLI entry point
```

## 🛠️ Technology Stack

- **Python 3.12+**
- **LangChain**: Document processing and LLM orchestration
- **Google Gemini**: LLM and embeddings
- **Pinecone**: Vector database for semantic search
- **Streamlit**: Web interface
- **Pydantic**: Data validation and configuration
- **PyPDF**: PDF text extraction

## 📋 Prerequisites

- Python 3.12 or higher
- Google AI API key (Gemini)
- Pinecone account and API key

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Project-3-Resume_to_Job_Search_Engine
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_INDEX_NAME=hireflow
PINECONE_DIMENSION=768
PINECONE_METRIC=cosine

# LLM Configuration
LLM_MODEL=gemini-2.5-flash

# Processing
MAX_TEXT_LENGTH=4000

# Logging
LOG_LEVEL=INFO
```

### 4. Run the Application

#### Web Interface (Recommended)
```bash
cd streamlit
streamlit run app.py
```

#### CLI Interface
```bash
python main.py
```

## 📚 Usage

### Web Interface

1. **Dashboard**: Overview of system status and document counts
2. **Upload Documents**: Add resumes and job descriptions
3. **Search Candidates**: Find matching candidates for specific roles
4. **Evaluate Matches**: Get AI-powered insights on candidate fit

### CLI Pipeline

The CLI provides a complete RAG pipeline:

```python
from main import LangChainHireFlowPipeline

pipeline = LangChainHireFlowPipeline()
pipeline.load_and_parse_documents()
pipeline.index_documents()
candidates = pipeline.search_candidates(job_description)
evaluations = pipeline.evaluate_candidates(job_description, candidates)
```

## 🔧 Configuration

Key configuration options in `utils/config.py`:

- **Vector Database**: Pinecone index settings
- **LLM Models**: Google Gemini model selection
- **Text Processing**: Maximum text length limits
- **Logging**: Log level configuration

## 📊 Data Models

### Resume Schema
```python
class Resume(BaseModel):
    candidate_id: str
    name: str
    email: EmailStr
    phone: Optional[str]
    location: Optional[str]
    skills: List[str]
    experience: Experience
    education: Education
    summary: Optional[str]
```

### Job Description Schema
```python
class JobDescription(BaseModel):
    jd_id: str
    title: str
    company: str
    location: str
    required_skills: List[str]
    optional_skills: List[str]
    experience_years: int
    description: str
```

### Candidate Evaluation Schema
```python
class CandidateEvaluation(BaseModel):
    candidate_id: str
    fit_score: float
    strengths: List[str]
    gaps: List[str]
    risks: Optional[List[str]]
    summary: str
```

## 🧪 Testing

Run the built-in system tests:

```bash
python test_system.py
```

## 🔍 How It Works

1. **Document Ingestion**: PDFs are loaded and text is extracted
2. **Text Processing**: Documents are split into chunks and cleaned
3. **LLM Parsing**: Structured data is extracted using Google Gemini
4. **Vector Embedding**: Text chunks are converted to embeddings
5. **Indexing**: Embeddings are stored in Pinecone vector database
6. **Semantic Search**: Queries find similar documents based on meaning
7. **AI Evaluation**: LLM generates candidate fit assessments
8. **Results**: Ranked candidates with explanations and scores

## 🚧 Current Status

✅ **Implemented:**
- Core RAG pipeline with LangChain
- LLM-based document parsing
- Pinecone vector indexing
- Semantic search and retrieval
- Candidate evaluation with AI
- Streamlit web interface
- Unified logging system
- Clean, maintainable codebase