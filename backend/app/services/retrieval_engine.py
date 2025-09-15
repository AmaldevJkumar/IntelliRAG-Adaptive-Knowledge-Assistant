"""
Advanced retrieval engine for RAG system
Multi-vector search with reranking and quality assessment
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from app.models.schemas import QueryType, RetrievedDocument
from app.config import get_settings
from app.utils.metrics import increment_counter, track_retrieval_time

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Advanced retrieval engine with multi-vector search capabilities
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = None
        self.reranker = None
        self.bm25_retriever = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and reranking models"""
        try:
            logger.info("Initializing retrieval models...")
            
            # Initialize embedding model
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL
            )
            
            # Initialize reranking model
            self.reranker = CrossEncoder(self.settings.RERANK_MODEL)
            
            logger.info("Retrieval models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval models: {e}")
            raise
    
    @track_retrieval_time
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_type: QueryType = QueryType.HYBRID,
        metadata_filters: Optional[Dict[str, Any]] = None,
        vector_store = None
    ) -> List[RetrievedDocument]:
        """
        Advanced multi-vector retrieval with hybrid search
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            query_type: Type of search (semantic, hybrid, keyword)
            metadata_filters: Optional metadata filters
            vector_store: Vector database instance
            
        Returns:
            List of retrieved documents with relevance scores
        """
        try:
            logger.debug(f"Starting retrieval: query_type={query_type}, top_k={top_k}")
            
            if query_type == QueryType.SEMANTIC:
                documents = await self._semantic_search(query, top_k, vector_store, metadata_filters)
            elif query_type == QueryType.KEYWORD:
                documents = await self._keyword_search(query, top_k, metadata_filters)
            elif query_type == QueryType.HYBRID:
                documents = await self._hybrid_search(query, top_k, vector_store, metadata_filters)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            # Apply post-processing
            documents = await self._post_process_results(query, documents)
            
            increment_counter("retrieval_requests_total", 
                            {"query_type": query_type.value, "top_k": str(top_k)})
            
            logger.debug(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            increment_counter("retrieval_errors_total", {"error_type": type(e).__name__})
            raise
    
    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        vector_store,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Perform semantic vector search"""
        try:
            logger.debug("Performing semantic search")
            
            # Generate query embedding
            query_embedding = await self._get_query_embedding(query)
            
            # Search vector store
            if vector_store is None:
                # Mock implementation for demo
                return await self._mock_semantic_search(query, top_k)
            
            # Real vector search implementation
            results = await vector_store.similarity_search_with_score(
                query_embedding=query_embedding,
                k=top_k,
                filter=metadata_filters
            )
            
            # Convert to RetrievedDocument format
            documents = []
            for i, (doc, score) in enumerate(results):
                documents.append(RetrievedDocument(
                    document_id=doc.metadata.get("document_id", f"doc_{i}"),
                    chunk_id=doc.metadata.get("chunk_id", f"chunk_{i}"),
                    content=doc.page_content,
                    relevance_score=float(score),
                    chunk_index=doc.metadata.get("chunk_index", i),
                    filename=doc.metadata.get("filename", "unknown.pdf"),
                    source=doc.metadata.get("source", "unknown"),
                    metadata=doc.metadata
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Perform keyword-based BM25 search"""
        try:
            logger.debug("Performing keyword search")
            
            # Mock implementation for demo
            return await self._mock_keyword_search(query, top_k)
            
            # Real BM25 implementation would go here
            # if self.bm25_retriever is None:
            #     await self._initialize_bm25_retriever()
            # 
            # results = self.bm25_retriever.get_relevant_documents(query)
            # return self._format_bm25_results(results[:top_k])
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise
    
    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        vector_store,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Perform hybrid semantic + keyword search with score fusion"""
        try:
            logger.debug("Performing hybrid search")
            
            # Get semantic and keyword results in parallel
            semantic_docs, keyword_docs = await asyncio.gather(
                self._semantic_search(query, top_k * 2, vector_store, metadata_filters),
                self._keyword_search(query, top_k * 2, metadata_filters)
            )
            
            # Fusion using Reciprocal Rank Fusion (RRF)
            fused_docs = await self._reciprocal_rank_fusion(
                semantic_docs, 
                keyword_docs, 
                alpha=self.settings.HYBRID_SEARCH_ALPHA
            )
            
            return fused_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    async def _reciprocal_rank_fusion(
        self,
        semantic_docs: List[RetrievedDocument],
        keyword_docs: List[RetrievedDocument],
        alpha: float = 0.5,
        k: int = 60
    ) -> List[RetrievedDocument]:
        """Combine results using Reciprocal Rank Fusion"""
        try:
            # Create score maps
            semantic_scores = {doc.chunk_id: 1.0 / (k + i) for i, doc in enumerate(semantic_docs)}
            keyword_scores = {doc.chunk_id: 1.0 / (k + i) for i, doc in enumerate(keyword_docs)}
            
            # Combine documents
            all_docs = {doc.chunk_id: doc for doc in semantic_docs + keyword_docs}
            
            # Calculate fused scores
            fused_scores = {}
            for chunk_id in all_docs.keys():
                semantic_score = semantic_scores.get(chunk_id, 0.0)
                keyword_score = keyword_scores.get(chunk_id, 0.0)
                fused_scores[chunk_id] = alpha * semantic_score + (1 - alpha) * keyword_score
            
            # Sort by fused score
            sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
            
            # Return sorted documents with updated scores
            result_docs = []
            for chunk_id in sorted_chunk_ids:
                doc = all_docs[chunk_id]
                doc.relevance_score = fused_scores[chunk_id]
                result_docs.append(doc)
            
            return result_docs
            
        except Exception as e:
            logger.error(f"Reciprocal rank fusion failed: {e}")
            raise
    
    async def rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int
    ) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder model"""
        try:
            logger.debug(f"Reranking {len(documents)} documents")
            
            if len(documents) <= top_k:
                return documents
            
            # Prepare query-document pairs
            pairs = [(query, doc.content) for doc in documents]
            
            # Get reranking scores
            try:
                rerank_scores = self.reranker.predict(pairs)
            except Exception as e:
                logger.warning(f"Reranking failed, using original scores: {e}")
                return documents[:top_k]
            
            # Update document scores and sort
            for doc, score in zip(documents, rerank_scores):
                doc.relevance_score = float(score)
                doc.metadata["reranked"] = True
                doc.metadata["original_score"] = doc.relevance_score
            
            # Sort by new scores
            reranked_docs = sorted(documents, key=lambda x: x.relevance_score, reverse=True)
            
            increment_counter("reranking_operations_total")
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            raise
    
    async def _post_process_results(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Post-process retrieval results"""
        try:
            # Remove duplicates
            seen_content = set()
            unique_docs = []
            
            for doc in documents:
                content_hash = hash(doc.content[:200])  # Use first 200 chars for dedup
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            # Apply quality filters
            filtered_docs = [
                doc for doc in unique_docs 
                if doc.relevance_score >= self.settings.RETRIEVAL_SCORE_THRESHOLD
            ]
            
            # Add query matching metadata
            for doc in filtered_docs:
                doc.metadata["query_terms_matched"] = await self._count_query_terms_matched(query, doc.content)
                doc.metadata["content_length"] = len(doc.content)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return documents
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            # Enhanced query preprocessing
            processed_query = await self._preprocess_query(query)
            
            # Generate embedding
            embedding = self.embeddings.embed_query(processed_query)
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise
    
    async def _preprocess_query(self, query: str) -> str:
        """Preprocess and enhance query"""
        try:
            # Basic preprocessing
            processed = query.strip().lower()
            
            # Remove special characters but keep important punctuation
            import re
            processed = re.sub(r'[^\w\s\?\!\.]', ' ', processed)
            
            # Query expansion could go here
            # - Add synonyms
            # - Add related terms
            # - Fix spelling
            
            return processed
            
        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            return query
    
    async def _count_query_terms_matched(self, query: str, content: str) -> int:
        """Count how many query terms appear in content"""
        try:
            query_terms = set(query.lower().split())
            content_terms = set(content.lower().split())
            return len(query_terms.intersection(content_terms))
            
        except Exception as e:
            logger.error(f"Query term matching failed: {e}")
            return 0
    
    async def get_query_suggestions(
        self,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """Get query suggestions based on partial input"""
        try:
            # Mock implementation - in practice, this would use:
            # - Historical query patterns
            # - Document content analysis
            # - User behavior data
            
            suggestions = [
                f"{partial_query} in machine learning",
                f"{partial_query} best practices",
                f"{partial_query} implementation guide",
                f"{partial_query} use cases",
                f"{partial_query} comparison"
            ]
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Query suggestions failed: {e}")
            return []
    
    async def explain_retrieval(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> Dict[str, Any]:
        """Explain why specific documents were retrieved"""
        try:
            explanation = {
                "query": query,
                "retrieval_strategy": "hybrid_search",
                "total_documents": len(documents),
                "explanation": {
                    "query_analysis": {
                        "terms": query.split(),
                        "length": len(query),
                        "type": "factual_question" if "?" in query else "keyword_search"
                    },
                    "document_matching": []
                }
            }
            
            for i, doc in enumerate(documents[:3]):  # Explain top 3
                doc_explanation = {
                    "rank": i + 1,
                    "document": doc.filename,
                    "relevance_score": doc.relevance_score,
                    "matching_factors": [
                        f"Semantic similarity: {doc.relevance_score:.3f}",
                        f"Query terms matched: {doc.metadata.get('query_terms_matched', 0)}",
                        f"Content relevance: High" if doc.relevance_score > 0.8 else "Content relevance: Medium"
                    ]
                }
                explanation["explanation"]["document_matching"].append(doc_explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Retrieval explanation failed: {e}")
            return {"error": str(e)}
    
    # Mock implementations for demo purposes
    async def _mock_semantic_search(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Mock semantic search for demo"""
        documents = []
        
        for i in range(min(top_k, 5)):
            documents.append(RetrievedDocument(
                document_id=f"doc_semantic_{i:03d}",
                chunk_id=f"chunk_semantic_{i:03d}",
                content=f"This semantic search result {i+1} contains relevant information about {query}. The content discusses advanced concepts, methodologies, and practical applications related to the user's query with high semantic similarity.",
                relevance_score=0.95 - (i * 0.05),
                chunk_index=i,
                filename=f"semantic_doc_{i+1}.pdf",
                source="semantic_search",
                metadata={
                    "search_type": "semantic",
                    "embedding_model": self.settings.EMBEDDING_MODEL,
                    "processing_time": 0.05 + (i * 0.01)
                }
            ))
        
        return documents
    
    async def _mock_keyword_search(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Mock keyword search for demo"""
        documents = []
        
        for i in range(min(top_k, 5)):
            documents.append(RetrievedDocument(
                document_id=f"doc_keyword_{i:03d}",
                chunk_id=f"chunk_keyword_{i:03d}",
                content=f"This keyword search result {i+1} contains exact term matches for '{query}'. The document includes specific terminology and concepts that directly correspond to the query terms with high lexical overlap.",
                relevance_score=0.85 - (i * 0.08),
                chunk_index=i,
                filename=f"keyword_doc_{i+1}.pdf",
                source="keyword_search",
                metadata={
                    "search_type": "keyword",
                    "algorithm": "BM25",
                    "term_frequency": 3 - i
                }
            ))
        
        return documents
