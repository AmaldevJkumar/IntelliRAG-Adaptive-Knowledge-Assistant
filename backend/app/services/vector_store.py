"""
Vector store service for RAG system
Multi-provider vector database abstraction
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from app.config import get_settings
from app.utils.metrics import track_vector_operation
from app.models.schemas import RetrievedDocument

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar vectors with scores"""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str]
    ) -> bool:
        """Add documents with embeddings"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        self.config = config
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            pinecone.init(
                api_key=self.config["api_key"],
                environment=self.config["environment"]
            )
            
            # Check if index exists, create if not
            index_name = self.config["index_name"]
            if index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=self.config["dimension"],
                    metric=self.config["metric"]
                )
            
            self.index = pinecone.Index(index_name)
            logger.info("Pinecone vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    @track_vector_operation
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search Pinecone for similar vectors"""
        try:
            # Convert to numpy array if needed
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            # Perform search
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True,
                filter=filter
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                doc_data = {
                    "page_content": match.metadata.get("content", ""),
                    "metadata": match.metadata
                }
                formatted_results.append((doc_data, float(match.score)))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Pinecone similarity search failed: {e}")
            raise
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str]
    ) -> bool:
        """Add documents to Pinecone"""
        try:
            # Prepare vectors for upsert
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vectors.append({
                    "id": ids[i],
                    "values": embedding,
                    "metadata": {
                        "content": doc["content"],
                        "filename": doc.get("filename", ""),
                        "source": doc.get("source", ""),
                        "chunk_id": doc.get("chunk_id", ""),
                        "document_id": doc.get("document_id", "")
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from Pinecone"""
        try:
            # Delete by filter (all chunks of the document)
            self.index.delete(filter={"document_id": document_id})
            logger.info(f"Deleted document {document_id} from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document from Pinecone: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": getattr(stats, 'index_fullness', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {}


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.config = config
        self.index = None
        self.metadata_store = {}  # Simple in-memory metadata store
        self.dimension = None
        self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        try:
            import os
            index_path = self.config["index_path"]
            
            if os.path.exists(f"{index_path}.index"):
                # Load existing index
                self.index = faiss.read_index(f"{index_path}.index")
                self.dimension = self.index.d
                
                # Load metadata if available
                if os.path.exists(f"{index_path}.metadata"):
                    import pickle
                    with open(f"{index_path}.metadata", 'rb') as f:
                        self.metadata_store = pickle.load(f)
                
                logger.info("Loaded existing FAISS index")
            else:
                # Create new index (will be initialized when first document is added)
                logger.info("FAISS index will be created when first document is added")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    def _create_index(self, dimension: int):
        """Create new FAISS index"""
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.dimension = dimension
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    @track_vector_operation
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search FAISS for similar vectors"""
        try:
            if self.index is None:
                logger.warning("FAISS index not initialized")
                return []
            
            # Convert to numpy array and normalize for cosine similarity
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and str(idx) in self.metadata_store:
                    metadata = self.metadata_store[str(idx)]
                    
                    # Apply filter if provided
                    if filter:
                        if not self._matches_filter(metadata, filter):
                            continue
                    
                    doc_data = {
                        "page_content": metadata.get("content", ""),
                        "metadata": metadata
                    }
                    results.append((doc_data, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            raise
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str]
    ) -> bool:
        """Add documents to FAISS"""
        try:
            if not embeddings:
                return True
            
            # Initialize index if needed
            if self.index is None:
                self._create_index(len(embeddings[0]))
            
            # Convert embeddings to numpy array and normalize
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)
            
            # Add vectors to index
            start_idx = self.index.ntotal
            self.index.add(vectors)
            
            # Store metadata
            for i, (doc, doc_id) in enumerate(zip(documents, ids)):
                metadata = {
                    "content": doc["content"],
                    "filename": doc.get("filename", ""),
                    "source": doc.get("source", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "document_id": doc.get("document_id", "")
                }
                self.metadata_store[str(start_idx + i)] = metadata
            
            # Save index and metadata
            await self._save_index()
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            return False
    
    async def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_path = self.config["index_path"]
            
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, f"{index_path}.index")
            
            # Save metadata
            import pickle
            with open(f"{index_path}.metadata", 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from FAISS (marks as deleted in metadata)"""
        try:
            # FAISS doesn't support direct deletion, so we mark as deleted
            deleted_count = 0
            for idx, metadata in self.metadata_store.items():
                if metadata.get("document_id") == document_id:
                    metadata["deleted"] = True
                    deleted_count += 1
            
            if deleted_count > 0:
                await self._save_index()
                logger.info(f"Marked {deleted_count} chunks as deleted for document {document_id}")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete document from FAISS: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        try:
            if self.index is None:
                return {"total_vectors": 0, "dimension": 0}
            
            active_vectors = sum(1 for meta in self.metadata_store.values() 
                               if not meta.get("deleted", False))
            
            return {
                "total_vectors": self.index.ntotal,
                "active_vectors": active_vectors,
                "dimension": self.dimension or 0,
                "index_type": type(self.index).__name__
            }
            
        except Exception as e:
            logger.error(f"Failed to get FAISS stats: {e}")
            return {}


class MockVectorStore(VectorStore):
    """Mock vector store for development and testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents = []
        logger.info("Mock vector store initialized")
    
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Mock similarity search"""
        results = []
        
        for i in range(min(k, 5)):  # Return up to 5 mock results
            doc_data = {
                "page_content": f"Mock search result {i+1} for your query. This contains relevant information that would be retrieved from the vector database based on semantic similarity.",
                "metadata": {
                    "document_id": f"mock_doc_{i:03d}",
                    "chunk_id": f"mock_chunk_{i:03d}",
                    "filename": f"mock_document_{i+1}.pdf",
                    "source": "mock_source",
                    "chunk_index": i
                }
            }
            score = 0.9 - (i * 0.1)  # Decreasing relevance scores
            results.append((doc_data, score))
        
        return results
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: List[str]
    ) -> bool:
        """Mock document addition"""
        self.documents.extend(documents)
        logger.info(f"Mock: Added {len(documents)} documents")
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        """Mock document deletion"""
        logger.info(f"Mock: Deleted document {document_id}")
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Mock statistics"""
        return {
            "total_vectors": len(self.documents),
            "dimension": 384,
            "provider": "mock"
        }


# Factory function and service initialization
async def create_vector_store(config: Dict[str, Any]) -> VectorStore:
    """Create vector store based on configuration"""
    store_type = config.get("type", "mock").lower()
    
    if store_type == "pinecone":
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone not available, using mock store")
            return MockVectorStore(config)
        return PineconeVectorStore(config)
    
    elif store_type == "faiss":
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using mock store")
            return MockVectorStore(config)
        return FAISSVectorStore(config)
    
    else:
        logger.info("Using mock vector store")
        return MockVectorStore(config)


# Global vector store instance
_vector_store = None


async def get_vector_store() -> VectorStore:
    """Get the global vector store instance"""
    global _vector_store
    
    if _vector_store is None:
        settings = get_settings()
        config = settings.get_vector_db_config()
        _vector_store = await create_vector_store(config)
    
    return _vector_store


async def initialize_vector_store():
    """Initialize the vector store on startup"""
    await get_vector_store()
    logger.info("Vector store service initialized")
