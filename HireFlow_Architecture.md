# 🎯 HireFlow System Architecture

## System Overview
HireFlow is an AI-powered recruitment platform that combines semantic search with Generative AI to intelligently match candidates to job descriptions.

## 🏗️ Complete System Architecture

```mermaid
graph TB
    %% User Interface Layer
    subgraph UI ["🖥️ User Interface Layer"]
        STREAMLIT[Streamlit Web App]
        CLI[CLI Interface]
        API[API Layer]
    end

    %% Core Application Layer
    subgraph CORE ["🧠 Core Application Layer"]
        ROUTER[Search Router<br/>LangChain RunnableBranch]
        MEMORY[Memory RAG<br/>Conversation History]
        EVALUATOR[RAG Evaluator<br/>RAGAS Metrics]
    end

    %% Search & Indexing Layer
    subgraph SEARCH ["🔍 Search & Indexing Layer"]
        HYBRID[Hybrid Indexer<br/>BM25 + Vector]
        VECTOR[Vector Store<br/>Pinecone]
        MULTI[Multi-Query<br/>LLM Expansion]
    end

    %% AI & Processing Layer
    subgraph AI ["🤖 AI & Processing Layer"]
        RERANKER[Re-Ranker<br/>Gemini LLM]
        PARSERS[Document Parsers<br/>Resume + JD]
        FILTERS[Result Filters<br/>Skills, Location, Exp]
        INGESTION[Document Ingestion<br/>PDF Processing]
    end

    %% Infrastructure Layer
    subgraph INFRA ["🛠️ Infrastructure Layer"]
        CONFIG[Configuration<br/>Environment & Settings]
        EMBEDDINGS[Embeddings<br/>Google GenAI]
        UTILS[Utilities<br/>Logging & Helpers]
    end



    %% External Services
    subgraph EXTERNAL ["🌐 External Services"]
        PINECONE[Pinecone<br/>Vector Database]
        GEMINI[Google Gemini<br/>LLM Services]
        GOOGLE_EMB[Google Embeddings<br/>Text-to-Vector]
    end

    %% Data Flow
    subgraph DATA_FLOW ["📊 Data Flow"]
        PDF[PDF Documents]
        TEXT[Extracted Text]
        VECTORS[Vector Embeddings]
        RESULTS[Search Results]
        EVALUATIONS[Quality Metrics]
    end

    %% User Interface Connections
    STREAMLIT --> ROUTER
    CLI --> ROUTER
    API --> ROUTER

    %% Core Layer Connections
    ROUTER --> HYBRID
    ROUTER --> MEMORY
    ROUTER --> EVALUATOR
    MEMORY --> EVALUATOR

    %% Search Layer Connections
    HYBRID --> VECTOR
    HYBRID --> MULTI
    MULTI --> GEMINI

    %% AI Layer Connections
    RERANKER --> GEMINI
    PARSERS --> GEMINI
    INGESTION --> PDF
    INGESTION --> TEXT
    INGESTION --> VECTORS

    %% Infrastructure Connections
    VECTOR --> PINECONE
    EMBEDDINGS --> GOOGLE_EMB
    CONFIG --> VECTOR
    CONFIG --> RERANKER
    CONFIG --> EMBEDDINGS

    %% Data Flow Connections
    TEXT --> EMBEDDINGS
    EMBEDDINGS --> VECTORS
    VECTORS --> VECTOR
    VECTOR --> RESULTS
    RESULTS --> EVALUATOR
    EVALUATOR --> EVALUATIONS

    %% Styling
    classDef uiLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef coreLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef searchLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef aiLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef infraLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    classDef externalLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef dataLayer fill:#fafafa,stroke:#424242,stroke-width:2px

    class STREAMLIT,CLI,API uiLayer
    class ROUTER,MEMORY,EVALUATOR coreLayer
    class HYBRID,VECTOR,MULTI searchLayer
    class RERANKER,PARSERS,FILTERS,INGESTION aiLayer
    class CONFIG,EMBEDDINGS,UTILS infraLayer

    class PINECONE,GEMINI,GOOGLE_EMB externalLayer
    class PDF,TEXT,VECTORS,RESULTS,EVALUATIONS dataLayer
```

## 🔄 Detailed Data Flow Diagram

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant UI as 🖥️ Interface
    participant Router as 🔍 Search Router
    participant Hybrid as Hybrid Indexer
    participant Vector as 📚 Vector Store
    participant BM25 as 🔤 BM25 Engine
    participant LLM as 🤖 Gemini LLM
    participant Memory as 🧠 Memory RAG
    participant Evaluator as RAG Evaluator

    User->>UI: Upload Documents
    UI->>Hybrid: Index Documents
    Hybrid->>Vector: Store Vectors
    Hybrid->>BM25: Build Index
    
    User->>UI: Search Query
    UI->>Router: Route Search
    Router->>Router: Determine Strategy
    
    alt Shallow Search
        Router->>Vector: Vector Search
        Vector-->>Router: Results
    else Deep Search
        Router->>Hybrid: Hybrid Search
        Hybrid->>BM25: Keyword Search
        Hybrid->>Vector: Vector Search
        Hybrid->>Hybrid: Combine Scores
        Hybrid-->>Router: Combined Results
        Router->>LLM: Rerank Results
        LLM-->>Router: Reranked Results
    end
    
    Router->>Memory: Record Search
    Router-->>UI: Search Results
    UI-->>User: Display Results
    
    User->>UI: Evaluate Quality
    UI->>Evaluator: Evaluate RAG
    Evaluator->>Evaluator: Calculate Metrics
    Evaluator-->>UI: Quality Report
    UI-->>User: Evaluation Results
```

## 🏛️ Component Relationship Diagram

```mermaid
graph LR
    subgraph "Core Components"
        A[Hybrid Indexer]
        B[Search Router]
        C[Vector Store]
        D[Memory RAG]
    end
    
    subgraph "Supporting Components"
        E[Re-Ranker]
        F[Parsers]
        G[Filters]
        H[Ingestion]
    end
    
    subgraph "Infrastructure"
        I[Configuration]
        J[Embeddings]
        K[Prompts]
        L[Utils]
    end
    
    %% Core relationships
    A --> B
    A --> C
    B --> D
    B --> E
    
    %% Supporting relationships
    A --> F
    A --> G
    A --> H
    
    %% Infrastructure relationships
    C --> I
    E --> I
    F --> I
    H --> J
    E --> K
    
    %% Styling
    classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef support fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef infra fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class A,B,C,D core
    class E,F,G,H support
    class I,J,K,L infra
```

## 📊 System Metrics & Monitoring

```mermaid
graph TB
    subgraph "System Health"
        HEALTH[System Status]
        PERFORMANCE[Performance Metrics]
        ERRORS[Error Tracking]
    end
    
    subgraph "Search Analytics"
        QUERIES[Search Queries]
        RESULTS[Result Quality]
        USAGE[Usage Patterns]
    end
    
    subgraph "AI Performance"
        LLM_METRICS[LLM Response Times]
        ACCURACY[Search Accuracy]
        RERANKING[Reranking Quality]
    end
    
    subgraph "Infrastructure"
        VECTOR_STATUS[Vector Store Status]
        API_STATUS[External API Status]
        MEMORY_USAGE[Memory Usage]
    end
    
    HEALTH --> PERFORMANCE
    HEALTH --> ERRORS
    QUERIES --> RESULTS
    RESULTS --> USAGE
    LLM_METRICS --> ACCURACY
    ACCURACY --> RERANKING
    VECTOR_STATUS --> API_STATUS
    API_STATUS --> MEMORY_USAGE
```

## 🎯 Key Features & Capabilities

- **🔍 Hybrid Search:** Combines BM25 keyword search with vector similarity

- **🤖 AI-Powered:** LLM-based reranking and evaluation with hardcoded prompts
- **📊 Quality Assessment:** RAGAS-based system evaluation
- **🧠 Memory System:** Persistent search history and analytics
- **🔄 Multi-Query:** Intelligent query expansion for better recall
- **📱 Dual Interface:** Web (Streamlit) + CLI interfaces
- **⚡ Scalable:** Modular architecture for easy extension
- **📚 Teaching-Friendly:** Clean, well-documented code structure



## 💡 **Simplified Design Philosophy**

This architecture follows the **KISS principle** (Keep It Simple, Stupid) by:

- **🎯 Direct Integration:** Components use hardcoded prompts for simplicity
- **🔧 Minimal Dependencies:** No complex prompt management overhead
- **📚 Learning Focus:** Easy to understand and modify for educational purposes
- **⚡ Performance:** Direct LLM calls without caching layers
- **🛠️ Maintainability:** Simple, straightforward code structure

This architecture provides a **robust, scalable, and maintainable** foundation for AI-powered recruitment, perfectly suited for both **production deployment** and **educational purposes**.
