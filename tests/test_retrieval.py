"""
Tests for retrieval engine functionality
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.retrieval_engine import RetrievalEngine
from backend.app.models.schemas import QueryType, RetrievedDocument


class TestRetrievalEngine:
    """Test retrieval engine functionality"""
    
    @pytest.fixture
    def retrieval_engine(self):
        """Create retrieval engine instance for testing"""
        with patch('backend.app.services.retrieval_engine.SentenceTransformerEmbeddings'):
            with patch('backend.app.services.retrieval_engine.CrossEncoder'):
                engine = RetrievalEngine()
                return engine
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, retrieval_engine, mock_vector_store):
        """Test semantic search functionality"""
        query = "What is machine learning?"
        top_k = 5
        
        # Mock the vector store
        with patch.object(retrieval_engine, '_get_query_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            documents = await retrieval_engine._semantic_search(
                query, top_k, mock_vector_store
            )
        
        assert len(documents) <= top_k
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)
        assert all(0 <= doc.relevance_score <= 1 for doc in documents)
    
    @pytest.mark.asyncio
    async def test_keyword_search(self, retrieval_engine):
        """Test keyword search functionality"""
        query = "machine learning algorithms"
        top_k = 3
        
        documents = await retrieval_engine._keyword_search(query, top_k)
        
        assert len(documents) <= top_k
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, retrieval_engine, mock_vector_store):
        """Test hybrid search combining semantic and keyword"""
        query = "deep learning neural networks"
        top_k = 5
        
        with patch.object(retrieval_engine, '_get_query_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            documents = await retrieval_engine._hybrid_search(
                query, top_k, mock_vector_store
            )
        
        assert len(documents) <= top_k
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)
    
    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, retrieval_engine):
        """Test reciprocal rank fusion algorithm"""
        semantic_docs = [
            RetrievedDocument(
                document_id="doc1", chunk_id="chunk1", content="Content 1",
                relevance_score=0.9, chunk_index=0, filename="doc1.pdf"
            ),
            RetrievedDocument(
                document_id="doc2", chunk_id="chunk2", content="Content 2", 
                relevance_score=0.8, chunk_index=1, filename="doc2.pdf"
            )
        ]
        
        keyword_docs = [
            RetrievedDocument(
                document_id="doc2", chunk_id="chunk2", content="Content 2",
                relevance_score=0.7, chunk_index=1, filename="doc2.pdf"
            ),
            RetrievedDocument(
                document_id="doc3", chunk_id="chunk3", content="Content 3",
                relevance_score=0.6, chunk_index=2, filename="doc3.pdf"
            )
        ]
        
        fused_docs = await retrieval_engine._reciprocal_rank_fusion(
            semantic_docs, keyword_docs, alpha=0.5
        )
        
        # Should have unique documents
        chunk_ids = [doc.chunk_id for doc in fused_docs]
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Should be sorted by fused score
        scores = [doc.relevance_score for doc in fused_docs]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_rerank_documents(self, retrieval_engine):
        """Test document reranking"""
        query = "artificial intelligence"
        documents = [
            RetrievedDocument(
                document_id=f"doc{i}", chunk_id=f"chunk{i}", 
                content=f"Content {i}", relevance_score=0.8 - (i * 0.1),
                chunk_index=i, filename=f"doc{i}.pdf"
            )
            for i in range(1, 6)
        ]
        
        # Mock the reranker
        with patch.object(retrieval_engine, 'reranker') as mock_reranker:
            mock_reranker.predict.return_value = [0.9, 0.7, 0.8, 0.6, 0.5]
            
            reranked_docs = await retrieval_engine.rerank_documents(
                query, documents, top_k=3
            )
        
        assert len(reranked_docs) == 3
        # Check that documents are sorted by new scores
        scores = [doc.relevance_score for doc in reranked_docs]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_post_process_results(self, retrieval_engine):
        """Test post-processing of retrieval results"""
        query = "test query"
        documents = [
            RetrievedDocument(
                document_id=f"doc{i}", chunk_id=f"chunk{i}",
                content="This is test content for processing.",
                relevance_score=0.8, chunk_index=i, filename=f"doc{i}.pdf"
            )
            for i in range(3)
        ]
        
        processed_docs = await retrieval_engine._post_process_results(query, documents)
        
        assert len(processed_docs) <= len(documents)
        assert all("query_terms_matched" in doc.metadata for doc in processed_docs)
        assert all("content_length" in doc.metadata for doc in processed_docs)
    
    @pytest.mark.asyncio
    async def test_query_preprocessing(self, retrieval_engine):
        """Test query preprocessing"""
        raw_query = "  What IS machine-learning?! "
        processed = await retrieval_engine._preprocess_query(raw_query)
        
        assert processed.strip() == processed
        assert processed.lower() == processed
        assert len(processed) > 0
    
    @pytest.mark.asyncio
    async def test_query_suggestions(self, retrieval_engine):
        """Test query suggestions generation"""
        partial_query = "machine"
        suggestions = await retrieval_engine.get_query_suggestions(partial_query, limit=5)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        assert all(partial_query in suggestion for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_explain_retrieval(self, retrieval_engine):
        """Test retrieval explanation"""
        query = "explain neural networks"
        documents = [
            RetrievedDocument(
                document_id="doc1", chunk_id="chunk1",
                content="Neural networks are computing systems inspired by biological neural networks.",
                relevance_score=0.9, chunk_index=0, filename="neural_networks.pdf",
                metadata={"query_terms_matched": 2}
            )
        ]
        
        explanation = await retrieval_engine.explain_retrieval(query, documents)
        
        assert "query" in explanation
        assert "explanation" in explanation
        assert "query_analysis" in explanation["explanation"]
        assert "document_matching" in explanation["explanation"]


class TestRetrievalIntegration:
    """Integration tests for retrieval engine"""
    
    @pytest.fixture
    def retrieval_engine(self):
        """Create retrieval engine with mocked dependencies"""
        with patch('backend.app.services.retrieval_engine.SentenceTransformerEmbeddings'):
            with patch('backend.app.services.retrieval_engine.CrossEncoder'):
                engine = RetrievalEngine()
                # Mock the embedding model
                engine.embeddings = MagicMock()
                engine.embeddings.embed_query.return_value = [0.1] * 384
                return engine
    
    @pytest.mark.asyncio
    async def test_retrieve_semantic(self, retrieval_engine, mock_vector_store):
        """Test semantic retrieval flow"""
        query = "machine learning fundamentals"
        
        documents = await retrieval_engine.retrieve(
            query=query,
            top_k=5,
            query_type=QueryType.SEMANTIC,
            vector_store=mock_vector_store
        )
        
        assert len(documents) > 0
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)
        
        # Check that vector store was called
        mock_vector_store.similarity_search_with_score.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_keyword(self, retrieval_engine):
        """Test keyword retrieval flow"""
        query = "artificial intelligence applications"
        
        documents = await retrieval_engine.retrieve(
            query=query,
            top_k=3,
            query_type=QueryType.KEYWORD
        )
        
        assert len(documents) > 0
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)
    
    @pytest.mark.asyncio
    async def test_retrieve_hybrid(self, retrieval_engine, mock_vector_store):
        """Test hybrid retrieval flow"""
        query = "deep learning convolutional networks"
        
        documents = await retrieval_engine.retrieve(
            query=query,
            top_k=5,
            query_type=QueryType.HYBRID,
            vector_store=mock_vector_store
        )
        
        assert len(documents) > 0
        assert all(isinstance(doc, RetrievedDocument) for doc in documents)
    
    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, retrieval_engine, mock_vector_store):
        """Test retrieval with metadata filters"""
        query = "machine learning"
        filters = {"source": "academic_papers", "year": 2024}
        
        documents = await retrieval_engine.retrieve(
            query=query,
            top_k=5,
            query_type=QueryType.SEMANTIC,
            metadata_filters=filters,
            vector_store=mock_vector_store
        )
        
        assert len(documents) > 0
        # Check that filters were passed to vector store
        mock_vector_store.similarity_search_with_score.assert_called_once()
        call_args = mock_vector_store.similarity_search_with_score.call_args
        assert call_args[1]["filter"] == filters
    
    @pytest.mark.asyncio
    async def test_retrieve_error_handling(self, retrieval_engine):
        """Test error handling in retrieval"""
        # Test with invalid query type
        with pytest.raises(ValueError):
            await retrieval_engine.retrieve(
                query="test",
                query_type="invalid_type"
            )


class TestRetrievalPerformance:
    """Performance tests for retrieval engine"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_retrieval_performance(self, retrieval_engine, mock_vector_store):
        """Test retrieval performance with multiple queries"""
        queries = [
            "machine learning fundamentals",
            "deep neural networks architecture", 
            "natural language processing techniques",
            "computer vision applications",
            "reinforcement learning algorithms"
        ]
        
        import time
        start_time = time.time()
        
        for query in queries:
            documents = await retrieval_engine.retrieve(
                query=query,
                top_k=5,
                query_type=QueryType.HYBRID,
                vector_store=mock_vector_store
            )
            assert len(documents) > 0
        
        total_time = time.time() - start_time
        avg_time = total_time / len(queries)
        
        # Should process each query in reasonable time
        assert avg_time < 1.0  # Less than 1 second per query
    
    @pytest.mark.asyncio
    async def test_large_result_handling(self, retrieval_engine, mock_vector_store):
        """Test handling of large result sets"""
        # Mock large result set
        large_results = []
        for i in range(100):
            large_results.append(({
                "page_content": f"Content {i}",
                "metadata": {"document_id": f"doc_{i}", "filename": f"file_{i}.pdf"}
            }, 0.9 - (i * 0.001)))
        
        mock_vector_store.similarity_search_with_score.return_value = large_results
        
        documents = await retrieval_engine.retrieve(
            query="test query",
            top_k=10,
            query_type=QueryType.SEMANTIC,
            vector_store=mock_vector_store
        )
        
        # Should limit results to top_k
        assert len(documents) == 10
        
        # Should be sorted by relevance
        scores = [doc.relevance_score for doc in documents]
        assert scores == sorted(scores, reverse=True)


class TestMockRetrievalMethods:
    """Test mock retrieval methods"""
    
    @pytest.fixture
    def retrieval_engine(self):
        """Create retrieval engine for mock testing"""
        with patch('backend.app.services.retrieval_engine.SentenceTransformerEmbeddings'):
            with patch('backend.app.services.retrieval_engine.CrossEncoder'):
                return RetrievalEngine()
    
    @pytest.mark.asyncio
    async def test_mock_semantic_search(self, retrieval_engine):
        """Test mock semantic search implementation"""
        query = "artificial intelligence"
        top_k = 3
        
        documents = await retrieval_engine._mock_semantic_search(query, top_k)
        
        assert len(documents) == min(top_k, 5)
        assert all(doc.source == "semantic_search" for doc in documents)
        assert all("semantic" in doc.document_id for doc in documents)
        assert all(query in doc.content for doc in documents)
    
    @pytest.mark.asyncio
    async def test_mock_keyword_search(self, retrieval_engine):
        """Test mock keyword search implementation"""
        query = "machine learning"
        top_k = 4
        
        documents = await retrieval_engine._mock_keyword_search(query, top_k)
        
        assert len(documents) == min(top_k, 5)
        assert all(doc.source == "keyword_search" for doc in documents)
        assert all("keyword" in doc.document_id for doc in documents)
        assert all(query in doc.content for doc in documents)
