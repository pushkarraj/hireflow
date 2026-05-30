"""
Pinecone vector database manager for semantic search.
Handles document embedding, indexing, and similarity search.
"""

import time
from typing import Dict, Any, List
from pinecone import Pinecone, ServerlessSpec       
from utils.utils import get_embeddings
from utils.config import (
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
)
from utils.utils import get_logger, is_quota_error

logger = get_logger(__name__)

class VectorStore:
    """Manages Pinecone vector database for semantic document search"""
    
    def __init__(self):
        """Initialize Pinecone client and components"""
        self.client = None       # Pinecone client instance
        self.index = None        # Pinecone index for vector operations
        self.embeddings = None   # Google embedding model
        self._ready = False      # Ready status flag
        
    def initialize(self) -> bool:
        """Set up Pinecone client, index, and embedding model"""
        try:
            if not PINECONE_API_KEY:
                logger.warning("PINECONE_API_KEY not set")
                return False
            
            # Create Pinecone client first
            self.client = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone client created successfully")
            
            # Ensure index exists
            if not self.ensure_index():
                logger.error("Failed to ensure index exists")
                return False
            
            # Get embeddings
            self.embeddings = get_embeddings()
            if not self.embeddings:
                logger.error("Failed to get embeddings")
                return False
            
            # Get the index
            self.index = self.client.Index(PINECONE_INDEX_NAME)
            logger.info("Vector store initialization completed successfully")
            
            self._ready = True
            return True
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            self._ready = False
            return False
    
    def ensure_index(self) -> bool:
        """Create Pinecone index if it doesn't exist or recreate if dimension mismatch"""
        try:
            if not self.client:
                logger.error("Pinecone client not initialized")
                return False
            
            # Check if index exists
            existing_indexes = [index.name for index in self.client.list_indexes()]
            if PINECONE_INDEX_NAME in existing_indexes:
                # Get index info
                index_info = self.client.describe_index(PINECONE_INDEX_NAME)
                current_dimension = index_info.dimension
                
                if current_dimension == PINECONE_DIMENSION:
                    logger.info(f"Index {PINECONE_INDEX_NAME} already exists with correct dimensions")
                    return True
                else:
                    logger.warning(f"Index dimension mismatch: {current_dimension} vs {PINECONE_DIMENSION}")
                    # Delete and recreate
                    self.client.delete_index(PINECONE_INDEX_NAME)
                    logger.info(f"Deleted index {PINECONE_INDEX_NAME} due to dimension mismatch")
                    time.sleep(5)  # Wait for deletion
            else:
                logger.info(f"Index {PINECONE_INDEX_NAME} not found, creating new one")
            
            # Create new index
            from pinecone import ServerlessSpec
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            self.client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=spec
            )
            logger.info(f"Created new index {PINECONE_INDEX_NAME}")
            time.sleep(10)  # Wait for index to be ready
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure index: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if all components are initialized and ready for use"""
        return self._ready
    
    def add_resumes(self, documents: List[Any]) -> bool:
        """Convert resume documents to embeddings and store in Pinecone"""
        if not self._ready or not self.index:
            return False
        
        try:
            # Prepare data for Pinecone
            vectors = []
            for i, doc in enumerate(documents):
                # Create metadata with page content
                metadata = doc.metadata.copy()
                metadata['page_content'] = doc.page_content
                metadata['type'] = 'resume'
                
                vectors.append({
                    "id": f"resume_{i}_{metadata.get('candidate_id', 'unknown')}",
                    "values": self.embeddings.embed_documents([doc.page_content])[0],
                    "metadata": metadata
                })
            
            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - indexing skipped")
            else:
                logger.error(f"Failed to add resumes: {e}")
            return False
    
    def add_job_descriptions(self, documents: List[Any]) -> bool:
        """Convert job description documents to embeddings and store in Pinecone"""
        if not self._ready or not self.index:
            return False
        
        try:
            # Prepare data for Pinecone
            vectors = []
            for i, doc in enumerate(documents):
                # Create metadata with page content
                metadata = doc.metadata.copy()
                metadata['page_content'] = doc.page_content
                metadata['type'] = 'job_description'
                
                vectors.append({
                    "id": f"jd_{i}_{metadata.get('jd_id', 'unknown')}",
                    "values": self.embeddings.embed_documents([doc.page_content])[0],
                    "metadata": metadata
                })
            
            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - indexing skipped")
            else:
                logger.error(f"Failed to add job descriptions: {e}")
            return False
    
    def search_resumes(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find similar resumes using semantic search with optional metadata filters"""
        if not self._ready or not self.index:
            return []
        
        try:
            # Prepare query vector
            query_vector = self.embeddings.embed_documents([query])[0]
            
            # Perform similarity search
            if filters:
                pinecone_filters = self.convert_filters_to_pinecone(filters)
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=pinecone_filters,
                    include_metadata=True
                )
            else:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
            
            # Check if results exist and have matches
            if not results or not hasattr(results, 'matches') or not results.matches:
                logger.warning(f"No results or matches found for query: {query}")
                return []
            
            return [
                {
                    'page_content': item.metadata.get('page_content', ''),
                    'metadata': item.metadata,
                    'score': item.score
                }
                for item in results.matches
            ]
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - search skipped")
            else:
                logger.error(f"Resume search failed: {e}")
            return []
    
    def search_job_descriptions(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find similar job descriptions using semantic search with optional metadata filters"""
        if not self._ready or not self.index:
            return []
        
        try:
            # Prepare query vector
            query_vector = self.embeddings.embed_documents([query])[0]
            
            # Perform similarity search
            if filters:
                pinecone_filters = self.convert_filters_to_pinecone(filters)
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=pinecone_filters,
                    include_metadata=True
                )
            else:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
            
            # Check if results exist and have matches
            if not results or not hasattr(results, 'matches') or not results.matches:
                logger.warning(f"No results or matches found for query: {query}")
                return []
            
            return [
                {
                    'page_content': item.metadata.get('page_content', ''),
                    'metadata': item.metadata,
                    'score': item.score
                }
                for item in results.matches
            ]
        except Exception as e:
            if is_quota_error(e):
                logger.warning("API quota exceeded - search skipped")
            else:
                logger.error(f"Job description search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics like vector count and dimensions"""
        if not self._ready or not self.index:
            return {"status": "not_ready"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "ready",
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "metric": stats.metric
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def convert_filters_to_pinecone(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Transform basic filters to Pinecone's query filter format"""
        pinecone_filters = {}
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                pinecone_filters[key] = {"$eq": value}
            elif isinstance(value, list):
                pinecone_filters[key] = {"$in": value}
            elif isinstance(value, dict):
                if "min" in value and "max" in value:
                    pinecone_filters[key] = {
                        "$gte": value["min"],
                        "$lte": value["max"]
                    }
                elif "min" in value:
                    pinecone_filters[key] = {"$gte": value["min"]}
                elif "max" in value:
                    pinecone_filters[key] = {"$lte": value["max"]}
        
        return pinecone_filters
