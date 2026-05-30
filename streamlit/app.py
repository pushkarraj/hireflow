"""HireFlow - Clean Architecture Implementation"""

import truststore
# Inject macOS Keychain certs so corporate SSL proxies (e.g. Zscaler) are trusted
truststore.inject_into_ssl()

import streamlit as st
from pathlib import Path
import sys
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.hybrid_indexer import HybridIndexer
from core.vector_store import VectorStore
from core.re_ranker import ReRanker
from core.ingestion import load_resumes, DocumentProcessor
from core.parsing import ResumeParser, JobParser
from utils.schemas import JobDescription, Resume
from utils.config import GOOGLE_API_KEY, LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from core.memory_rag import MemoryRAG

# Data directory
DATA_RESUMES_DIR = project_root / "data" / "resumes"

# Page config
st.set_page_config(page_title="HireFlow", page_icon="🎯", layout="wide")

# ============================================================================
# SYSTEM INITIALIZATION (Clean, minimal)
# ============================================================================

class SystemManager:
    """Manages system components without business logic"""
    
    def __init__(self):
        self._components = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize core system components"""
        if self._initialized:
            return True
            
        try:
            # Initialize VectorStore
            vector_store = VectorStore()
            vector_store_ready = vector_store.initialize()
            
            # Create LLM
            llm = None
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                )
            except Exception as e:
                st.warning(f"LLM initialization failed: {e}")
            
            # Create core components — pass the already-initialized VectorStore so
            # HybridIndexer reuses it instead of opening a second Pinecone connection.
            hybrid_indexer = HybridIndexer(vector_store=vector_store) if vector_store_ready else None
            document_processor = DocumentProcessor()
            resume_parser = ResumeParser()
            job_parser = JobParser()
            reranker = ReRanker()
            memory_rag = MemoryRAG()
            
            # Auto-index existing resumes using core module
            if hybrid_indexer and vector_store_ready:
                try:
                    existing_resumes = load_resumes(str(DATA_RESUMES_DIR))
                    if existing_resumes:
                        hybrid_indexer.index_resumes(existing_resumes)
                        st.success(f"Indexed {len(existing_resumes)} existing resumes")
                except Exception as e:
                    st.warning(f"Auto-indexing failed: {e}")
            
            # Store components
            self._components = {
                'vector_store': vector_store,
                'hybrid_indexer': hybrid_indexer,
                'llm': llm,
                'document_processor': document_processor,
                'resume_parser': resume_parser,
                'job_parser': job_parser,
                'reranker': reranker,
                'memory_rag': memory_rag,
                'vector_store_ready': vector_store_ready
            }
            
            self._initialized = True
            return True
            
        except Exception as e:
            st.error(f"System initialization failed: {e}")
            return False

    def get_component(self, name: str) -> Any:
        """Get component by name"""
        if not self._initialized:
            raise RuntimeError("System not initialized")
        return self._components.get(name)
    
    def is_ready(self) -> bool:
        """Check if system is ready"""
        return self._initialized and self._components.get('vector_store_ready', False)

# ============================================================================
# UI LAYER (Only UI logic, no business logic)
# ============================================================================

class HireFlowUI:
    """Clean UI layer that delegates to core modules"""
    
    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager
    
    def render_upload_section(self):
        """Render resume upload section"""
        st.header("Add Resumes")
        st.info("Upload PDF resumes to search through")
        
        resume_files = st.file_uploader("Select PDF Resumes", type="pdf", accept_multiple_files=True)
        
        if resume_files:
            st.success(f"Selected {len(resume_files)} resume(s)")
            if st.button("Process & Index Resumes", type="primary"):
                self.handle_resume_upload(resume_files)
    
    def render_search_section(self):
        """Render candidate search section"""
        st.header("Search Candidates")
        st.info("Enter job details to find matching candidates")
        
        with st.form("search_form"):
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Accountant")
            job_description = st.text_area("Job Description", placeholder="Enter detailed job requirements...", height=100)
            required_skills = st.text_area("Required Skills (one per line)", placeholder="Python\nJavaScript\nReact")
            top_k = st.slider("Number of Results", 3, 10, 5)
            
            submitted = st.form_submit_button("Find Candidates", type="primary")
        
        if submitted and job_description:
            self.handle_search(job_title, job_description, required_skills, top_k)
    
    def render_status_sidebar(self):
        """Render system status in sidebar"""
        st.sidebar.header("System Status")
        
        if self.system_manager.is_ready():
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            if hybrid_indexer:
                st.sidebar.write(f"**Resumes Indexed:** {len(hybrid_indexer.resume_texts)}")
            st.sidebar.write("**System:** Ready")
        else:
            st.sidebar.write("**System:** Initializing...")
        
        # Add Memory & Evaluation section
        st.sidebar.markdown("---")
        st.sidebar.write("**Memory & Evaluation:**")
        
        memory_rag = self.system_manager.get_component('memory_rag')
        if memory_rag:
            stats = memory_rag.get_memory_stats()
            st.sidebar.write(f"• Total Interactions: {stats['total_messages']}")
            st.sidebar.write(f"• Searches: {stats['search_count']}")
            st.sidebar.write(f"• Candidate Views: {stats['candidate_views']}")
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.write("**Navigation:**")
        if st.sidebar.button("📊 Memory & Evaluation", use_container_width=True):
            st.session_state.page = "memory_eval"
        if st.sidebar.button("🏠 Main Page", use_container_width=True):
            st.session_state.page = "main"
    
    def render_memory_evaluation_page(self):
        """Render Memory & Evaluation page"""
        st.header("🧠 Memory & Evaluation Dashboard")
        
        # Memory RAG Section
        st.subheader("📝 Search Memory")
        memory_rag = self.system_manager.get_component('memory_rag')
        
        if memory_rag:
            # Show memory stats
            col1, col2, col3 = st.columns(3)
            stats = memory_rag.get_memory_stats()
            
            with col1:
                st.metric("Total Interactions", stats['total_messages'])
            with col2:
                st.metric("Searches", stats['search_count'])
            with col3:
                st.metric("Candidate Views", stats['candidate_views'])
            
            # Show recent search history
            st.subheader("🔍 Recent Search History")
            search_history = memory_rag.get_search_history()
            if search_history:
                for i, query in enumerate(search_history, 1):
                    st.write(f"{i}. **{query}**")
            else:
                st.info("No search history yet. Perform some searches to see them here!")
            
            # Show recent interactions
            st.subheader("👥 Recent Interactions")
            messages = memory_rag.memory.chat_memory.messages[-10:]  # Last 10 messages
            for msg in messages:
                if hasattr(msg, 'content'):
                    if "Search:" in msg.content:
                        st.write(f"🔍 **{msg.content}**")
                    elif "Viewed candidate:" in msg.content:
                        st.write(f"👀 **{msg.content}**")
                    else:
                        st.write(f"💬 {msg.content}")
        else:
            st.warning("Memory RAG not available")
        
        # RAG Evaluation Section
        st.subheader("📊 RAG Quality Evaluation")
        st.info("Evaluate the quality of your search results using RAGAS metrics")
        
        # Simple evaluation form
        with st.form("evaluation_form"):
            eval_query = st.text_input("Query to Evaluate", placeholder="e.g., Senior Accountant with QuickBooks")
            expected_skills = st.text_area("Expected Skills (one per line)", placeholder="QuickBooks\nAccounting\nExcel")
            eval_top_k = st.slider("Number of Results to Evaluate", 3, 10, 5)
            
            if st.form_submit_button("Evaluate Search Quality"):
                if eval_query and expected_skills:
                    self.run_evaluation(eval_query, expected_skills.split('\n'), eval_top_k)
                else:
                    st.warning("Please provide both query and expected skills")
    
    def run_evaluation(self, query: str, expected_skills: list, top_k: int):
        """Run RAG evaluation"""
        try:
            from core.evaluator import RAGEvaluator
            
            evaluator = RAGEvaluator()
            st.info("Running evaluation... This may take a moment.")
            
            # Clean up skills list
            skills = [s.strip() for s in expected_skills if s.strip()]
            
            # Run evaluation
            metrics = evaluator.evaluate_search_quality(query, skills, "deep", top_k)
            
            if metrics:
                st.success("Evaluation completed!")
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Answer Relevancy", f"{metrics.answer_relevancy:.2f}")
                    st.metric("Context Precision", f"{metrics.context_precision:.2f}")
                
                with col2:
                    st.metric("Faithfulness", f"{metrics.faithfulness:.2f}")
                    st.metric("Answer Correctness", f"{metrics.answer_correctness:.2f}")
                
                st.metric("Overall Score", f"{metrics.overall_score:.2f}", delta=f"{metrics.overall_score - 5:.2f}")
                
                # Interpretation
                if metrics.overall_score >= 8:
                    st.success("Excellent search quality! 🎉")
                elif metrics.overall_score >= 6:
                    st.info("Good search quality. Room for improvement.")
                else:
                    st.warning("Search quality needs improvement. Consider refining your query or adding more relevant resumes.")
            else:
                st.warning("Evaluation returned no results. Check your query and available data.")
                
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.info("This might be due to missing dependencies or configuration issues.")
    
    def handle_resume_upload(self, resume_files):
        """Handle resume upload using core modules"""
        try:
            document_processor = self.system_manager.get_component('document_processor')
            resume_parser = self.system_manager.get_component('resume_parser')
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            
            if not all([document_processor, resume_parser, hybrid_indexer]):
                st.error("System not ready for resume processing")
                return
            
            st.info(f"Processing {len(resume_files)} resume(s)...")
            
            processed = 0
            for file in resume_files:
                try:
                    # Use core module to process resume
                    import os
                    os.makedirs(DATA_RESUMES_DIR, exist_ok=True)
                    temp_path = str(DATA_RESUMES_DIR / file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Use core DocumentProcessor to get text
                    text = document_processor.load_pdf(temp_path)
                    if text:
                        # Create proper Document object for indexing
                        from langchain_core.documents import Document
                        resume_doc = Document(
                            page_content=text,
                            metadata={
                                'source': temp_path,
                                'filename': file.name,
                                'candidate_id': f"c_{file.name.replace('.pdf', '')}",
                                'name': file.name.replace('.pdf', '').replace('_', ' ').title()
                            }
                        )
                        
                        # Use core ResumeParser for metadata extraction
                        candidate_id = f"c_{file.name.replace('.pdf', '')}"
                        resume_parser.parse_resume(text, candidate_id)
                        
                        # Use core HybridIndexer with proper Document object
                        indexing_result = hybrid_indexer.index_resumes([resume_doc])
                        
                        if indexing_result:
                            processed += 1
                            st.success(f"{file.name} - Indexed successfully")
                        else:
                            st.warning(f"{file.name} - Indexing failed")
                    else:
                        st.error(f"{file.name} - Could not extract text")
                        
                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")
            
            if processed > 0:
                st.success(f"Successfully processed and indexed {processed} resumes!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Resume upload failed: {e}")
    
    def handle_search(self, job_title: str, job_description: str, required_skills: str, top_k: int):
        """Handle candidate search using core modules"""
        try:
            hybrid_indexer = self.system_manager.get_component('hybrid_indexer')
            reranker = self.system_manager.get_component('reranker')
            
            if not hybrid_indexer:
                st.error("Search service not available")
                return
            
            with st.spinner("Searching candidates..."):
                # Use core HybridIndexer for search
                skills_list = [s.strip() for s in required_skills.split('\n') if s.strip()] if required_skills else []
                search_query = f"{job_title} {' '.join(skills_list)} {job_description}"
                
                # Record search in Memory RAG
                memory_rag = self.system_manager.get_component('memory_rag')
                if memory_rag:
                    memory_rag.record_search(search_query, 0)  # Will update with actual count
                
                # Core module handles the search logic
                candidates_data = hybrid_indexer.search_resumes(search_query, top_k=top_k)
                
                # Update search count in Memory RAG
                if memory_rag and candidates_data:
                    memory_rag.record_search(search_query, len(candidates_data))
                
                if candidates_data:
                    st.success(f"Found {len(candidates_data)} candidates!")

                    # Record candidate views in Memory RAG
                    if memory_rag:
                        for candidate in candidates_data[:3]:  # Record top 3 candidates
                            candidate_name = candidate.get('name', 'Unknown Candidate')
                            memory_rag.record_candidate_view(candidate_name)

                    self.display_search_results(candidates_data, job_title or "Position", job_description, reranker)
                else:
                    st.warning("No matching candidates found")
                    
        except Exception as e:
            st.error(f"Search failed: {e}")
    
    def display_search_results(self, candidates_data: list, job_title: str, job_description: str, reranker: ReRanker):
        """Display search results from core module data"""
        st.header(f"Top Matches for: {job_title}")
        
        for i, candidate in enumerate(candidates_data):
            with st.expander(f"{candidate.get('name', f'Candidate {i+1}')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Final Score:** {candidate.get('combined_score', 0):.2f}")
                    st.write(f"**Skills:** {', '.join(candidate.get('skills', []))}")
                    st.write(f"**Experience:** {candidate.get('experience', 'N/A')}")
                    st.write(f"**Location:** {candidate.get('location', 'N/A')}")
                    
                    if candidate.get('text'):
                        st.markdown("**Resume Preview:**")
                        text = candidate.get('text', '')
                        st.text(text[:300] + "..." if len(text) > 300 else text)
                    
                    # AI evaluation for top candidates using core ReRanker
                    if i < 3 and reranker:
                        self.display_ai_evaluation(candidate, job_title, job_description, reranker)
                
                with col2:
                    # Show only the final score
                    final_score = candidate.get('combined_score', 0)
                    
                    if final_score > 0:
                        st.metric("Final Score", f"{final_score:.2f}")
    
    def display_ai_evaluation(self, candidate: dict, job_title: str, job_description: str, reranker: ReRanker):
        """Display AI evaluation using core ReRanker"""
        try:
            # Create objects for core ReRanker
            jd = JobDescription(
                jd_id=f"jd_{job_title.lower().replace(' ', '_')}",
                title=job_title or "Position",
                text=job_description
            )
            
            resume = Resume(
                candidate_id=candidate.get('candidate_id', 'unknown'),
                name=candidate.get('name', 'Unknown'),
                email="candidate@example.com",
                phone="+1-555-0000",
                experience=5,
                skills=candidate.get('skills', []),
                education="Bachelor's Degree",
                text=candidate.get('text', '')
            )
            
            # Use core ReRanker for evaluation
            evaluation = reranker.evaluate_candidate(resume, jd)
            if evaluation:
                eval_data = evaluation.model_dump()
                fit_score = eval_data.get('fit_score', 0)
                
                st.markdown("**AI Evaluation:**")
                if fit_score >= 8:
                    st.success(f"**AI Score: {fit_score}/10**")
                elif fit_score >= 6:
                    st.info(f"**AI Score: {fit_score}/10**")
                else:
                    st.warning(f"**AI Score: {fit_score}/10**")
                
                # Show strengths and gaps from core evaluation
                strengths = eval_data.get('strengths', [])
                if strengths:
                    st.write("**Strengths:**")
                    for s in strengths[:2]:
                        st.write(f"• {s}")
                
                gaps = eval_data.get('gaps', [])
                if gaps:
                    st.write("**Gaps:**")
                    for g in gaps[:2]:
                        st.write(f"• {g}")
                
                summary = eval_data.get('summary', '')
                if summary:
                    st.write("**Summary:**")
                    st.write(summary)
                    
        except Exception as e:
            st.info("AI evaluation not available")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    st.title("HireFlow - AI Resume Search")

    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    # Initialize system manager
    system_manager = SystemManager()
    if not system_manager.initialize():
        st.error("System initialization failed. Please check configuration.")
        return

    # Create UI with dependency injection
    ui = HireFlowUI(system_manager)

    # Render status sidebar (always visible)
    ui.render_status_sidebar()

    # Render page based on navigation state
    if st.session_state.get('page') == 'memory_eval':
        ui.render_memory_evaluation_page()
    else:
        # Render main page UI components
        col1, col2 = st.columns([1, 1])

        with col1:
            ui.render_upload_section()

        with col2:
            ui.render_search_section()

if __name__ == "__main__":
    main()
