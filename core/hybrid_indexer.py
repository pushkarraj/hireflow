"""
Hybrid search system combining BM25 (lexical) and vector (semantic) search.
Provides comprehensive candidate and job matching capabilities.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from core.vector_store import VectorStore
from utils.utils import get_logger

logger = get_logger(__name__)

class HybridIndexer:
    """Combines BM25 keyword search with vector semantic search for better results"""

    def __init__(self, vector_store: VectorStore = None):
        """Initialize both BM25 and vector search components.

        Args:
            vector_store: Optional pre-initialized VectorStore to reuse.
                          If not provided a new one is created and initialized.
                          Pass the app-level instance to avoid a second Pinecone
                          connection being opened on startup.
        """
        if vector_store is not None:
            logger.info("HybridIndexer: reusing provided VectorStore instance")
            self.vector_store = vector_store
        else:
            logger.info("HybridIndexer: creating new VectorStore instance")
            self.vector_store = VectorStore()
            self.vector_store.initialize()  # Set up Pinecone vector store
        self.bm25_resumes = None        # BM25 index for resumes
        self.bm25_jds = None           # BM25 index for job descriptions
        self.resume_texts = []         # Lowercased text content for BM25 resume search
        self.resume_metadata = []      # Original metadata (name, candidate_id, etc.) per resume
        self.jd_texts = []             # Text content for BM25 JD search
        logger.info("HybridIndexer initialized")
    
    def index_resumes(self, resumes: List[Document]) -> bool:
        """Index resumes for both keyword and semantic search.

        Appends to existing index so multiple upload calls accumulate correctly.
        """
        if not resumes:
            logger.info("index_resumes: no documents provided, skipping")
            return False

        logger.info(f"index_resumes: indexing {len(resumes)} new resume(s)")
        try:
            # Append new resumes to existing BM25 corpus (do not reset)
            added = 0
            for resume in resumes:
                text = resume.page_content.lower()
                if text.strip():
                    self.resume_texts.append(text)
                    # Preserve original metadata alongside the lowercased text so
                    # combine_results can surface the real candidate name/id later.
                    self.resume_metadata.append(resume.metadata)
                    added += 1
                    logger.info(f"  BM25: added '{resume.metadata.get('name', resume.metadata.get('filename', '?'))}' (corpus size now {len(self.resume_texts)})")

            if self.resume_texts:
                # Rebuild BM25 over the full accumulated corpus
                tokenized_texts = [text.split() for text in self.resume_texts]
                self.bm25_resumes = BM25Okapi(tokenized_texts)
                logger.info(f"BM25 index rebuilt with {len(self.resume_texts)} total resume(s)")

            # Add to vector store if available
            if self.vector_store.is_ready():
                logger.info(f"Sending {len(resumes)} resume(s) to Pinecone vector store")
                self.vector_store.add_resumes(resumes)
            else:
                logger.warning("Vector store not ready — skipping Pinecone upsert for resumes")

            logger.info(f"index_resumes complete: {added} resume(s) added, total BM25 corpus={len(self.resume_texts)}")
            return True

        except Exception as e:
            logger.error(f"Resume indexing failed: {e}")
            return False
    
    def index_job_descriptions(self, job_descriptions: List[Document]) -> bool:
        """Index job descriptions for both keyword and semantic search"""
        if not job_descriptions:
            logger.info("index_job_descriptions: no documents provided, skipping")
            return False

        logger.info(f"index_job_descriptions: indexing {len(job_descriptions)} JD(s)")
        try:
            # Prepare texts for BM25
            self.jd_texts = []
            for jd in job_descriptions:
                text = jd.page_content.lower()
                if text.strip():
                    self.jd_texts.append(text)
                    logger.info(f"  BM25: added JD '{jd.metadata.get('title', jd.metadata.get('filename', '?'))}'")

            if self.jd_texts:
                # Build BM25 model
                tokenized_texts = [text.split() for text in self.jd_texts]
                self.bm25_jds = BM25Okapi(tokenized_texts)
                logger.info(f"BM25 JD index built with {len(self.jd_texts)} JD(s)")

            # Add to vector store if available
            if self.vector_store.is_ready():
                logger.info(f"Sending {len(job_descriptions)} JD(s) to Pinecone vector store")
                self.vector_store.add_job_descriptions(job_descriptions)
            else:
                logger.warning("Vector store not ready — skipping Pinecone upsert for JDs")

            logger.info(f"index_job_descriptions complete: {len(self.jd_texts)} JD(s) indexed")
            return True

        except Exception as e:
            logger.error(f"Job description indexing failed: {e}")
            return False
    
    def search_resumes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search resumes using both BM25 and vector search, then combine results"""
        if not self.bm25_resumes:
            logger.warning("search_resumes: BM25 index is empty — no resumes indexed yet")
            return []

        logger.info(f"search_resumes: query='{query[:80]}', top_k={top_k}, BM25 corpus={len(self.resume_texts)}")
        try:
            # BM25 search
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_resumes.get_scores(query_tokens)
            top_bm25 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
            logger.info(f"BM25 top scores: {[(self.resume_metadata[i].get('name','?'), round(s,3)) for i,s in top_bm25 if i < len(self.resume_metadata)]}")

            # Vector search (if available) - with type filter for resumes only
            vector_results = []
            if self.vector_store.is_ready():
                # Add filter to only get resume documents
                filters = {"type": "resume"}
                logger.info(f"Vector search: querying Pinecone with top_k={top_k * 2}")
                vector_results = self.vector_store.search_resumes(query, top_k * 2, filters)
                logger.info(f"Vector search returned {len(vector_results)} result(s)")
            else:
                logger.warning("Vector store not ready — using BM25 only")

            # Combine results
            combined = self.combine_results(bm25_scores, vector_results, top_k, is_jd=False)
            logger.info(f"search_resumes complete: returning {len(combined)} candidate(s)")
            return combined

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_job_descriptions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search job descriptions using both BM25 and vector search, then combine results"""
        if not self.bm25_jds:
            return []
        
        try:
            # BM25 search
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_jds.get_scores(query_tokens)
            
            # Vector search (if available) - with type filter for job descriptions only
            vector_results = []
            if self.vector_store.is_ready():
                # Add filter to only get job description documents
                filters = {"type": "job_description"}
                vector_results = self.vector_store.search_job_descriptions(query, top_k * 2, filters)
            
            # Combine results
            return self.combine_results(bm25_scores, vector_results, top_k, is_jd=True)
                
        except Exception as e:
            logger.error(f"JD search failed: {e}")
            return []
    
    def combine_results(self, bm25_scores: List[float], vector_results: List[Dict],
                        top_k: int, is_jd: bool) -> List[Dict[str, Any]]:
        """Merge BM25 keyword and vector semantic results into unified ranked list"""
        results = []

        # Get metadata based on type
        metadata_list = self.jd_texts if is_jd else self.resume_texts

        # Normalize BM25 scores to [0, 1] so they are comparable with vector
        # cosine similarity scores (which are already in [0, 1]).
        max_bm25 = max((float(s) for s in bm25_scores), default=1.0) or 1.0
        logger.info(f"combine_results: {len(bm25_scores)} BM25 score(s), {len(vector_results)} vector result(s), max_bm25={round(max_bm25,3)}")

        # Process BM25 results - create normalized results
        bm25_results = []
        for i, score in enumerate(bm25_scores):
            if i < len(metadata_list):
                normalized_score = float(score) / max_bm25
                # Create consistent metadata for BM25 results
                if is_jd:
                    bm25_results.append({
                        'jd_id': f"jd_{i}",
                        'text': self.jd_texts[i],
                        'title': f"Job {i+1}",
                        'combined_score': normalized_score,
                        'bm25_score': normalized_score,
                        'vector_score': 0.0,
                        'source': 'bm25'
                    })
                else:
                    # Use stored metadata to get the real candidate name and id
                    meta = self.resume_metadata[i] if i < len(self.resume_metadata) else {}
                    name = meta.get('name') or f"Candidate {i+1}"
                    candidate_id = meta.get('candidate_id') or f"c_{i}"

                    bm25_results.append({
                        'candidate_id': candidate_id,
                        'text': self.resume_texts[i],
                        'name': name,
                        'skills': meta.get('skills', []),
                        'location': meta.get('location', 'Unknown'),
                        'experience': meta.get('experience', 5),
                        'combined_score': normalized_score,
                        'bm25_score': normalized_score,
                        'vector_score': 0.0,
                        'source': 'bm25'
                    })
        
        # Process vector results - create normalized results  
        vector_formatted = []
        for vec_result in vector_results:
            if is_jd:
                jd_id = vec_result.get('metadata', {}).get('jd_id', f"jd_vec_{len(vector_formatted)}")
                vector_formatted.append({
                    'jd_id': jd_id,
                    'text': vec_result.get('page_content', ''),
                    'title': vec_result.get('metadata', {}).get('title', 'Unknown Job'),
                    'combined_score': vec_result.get('score', 0.0),
                    'bm25_score': 0.0,
                    'vector_score': vec_result.get('score', 0.0),
                    'source': 'vector'
                })
            else:
                # For resumes
                candidate_id = vec_result.get('metadata', {}).get('candidate_id', f"c_vec_{len(vector_formatted)}")
                vector_formatted.append({
                    'candidate_id': candidate_id,
                    'text': vec_result.get('page_content', ''),
                    'name': vec_result.get('metadata', {}).get('name', 'Unknown Candidate'),
                    'skills': vec_result.get('metadata', {}).get('skills', []),
                    'location': vec_result.get('metadata', {}).get('location', 'Unknown'),
                    'experience': vec_result.get('metadata', {}).get('experience', 5),
                    'combined_score': vec_result.get('score', 0.0),
                    'bm25_score': 0.0,
                    'vector_score': vec_result.get('score', 0.0),
                    'source': 'vector'
                })
        
        # Combine all results
        all_results = bm25_results + vector_formatted
        logger.info(f"combine_results: {len(bm25_results)} BM25 result(s) + {len(vector_formatted)} vector result(s) = {len(all_results)} total")

        # Sort by combined score and return top results
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)

        # Return top_k results, removing 'source' field
        for result in all_results[:top_k]:
            result.pop('source', None)
            results.append(result)

        logger.info(f"combine_results: returning top {len(results)} result(s): {[r.get('name', r.get('title','?')) for r in results]}")
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get status of BM25 and vector search components"""
        return {
            'resumes_ready': bool(self.bm25_resumes),
            'jds_ready': bool(self.bm25_jds),
            'vector_store_ready': self.vector_store.is_ready(),
            'hybrid_ready': bool(self.bm25_resumes) and self.vector_store.is_ready()
        }
