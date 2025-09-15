"""
Query processing endpoints for RAG system
Advanced retrieval and generation capabilities
"""

import logging
import asyncio
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

from app.models.schemas import (
    QueryRequest, QueryResponse, StreamingQueryResponse, 
    RetrievedDocument, FeedbackRequest, FeedbackResponse
)
from app.services.retrieval_engine import RetrievalEngine
from app.services.generation_service import GenerationService
from app.services.vector_store import get_vector_store
from app.services.monitoring import QualityMonitor, QueryEvent
from app.services.cache import get_cache
from app.utils.metrics import track_query_time, increment_counter
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
retrieval_engine = RetrievalEngine()
generation_service = GenerationService()
quality_monitor = QualityMonitor()
settings = get_settings()


@router.post("", response_model=QueryResponse)
@track_query_time
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """
    🔍 **Advanced RAG Query Processing**
    
    Processes user queries through a sophisticated RAG pipeline featuring:
    - Multi-stage retrieval with quality assessment
    - Hybrid search combining semantic and keyword approaches
    - Cross-encoder reranking for improved relevance
    - Confidence scoring and uncertainty quantification
    - Real-time performance monitoring and feedback integration
    
    **Process Flow:**
    1. Query preprocessing and enhancement
    2. Multi-vector retrieval from knowledge base
    3. Document reranking and quality filtering
    4. Context assembly and prompt engineering
    5. LLM generation with streaming support
    6. Response validation and confidence scoring
    7. Performance metrics collection and monitoring
    """
    query_id = uuid4()
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Processing query [{query_id}]: {request.query[:100]}...")
        
        # Increment query counter
        increment_counter("queries_total", {"query_type": request.query_type.value})
        
        # Check cache first
        cache = await get_cache()
        cache_key = f"query:{hash(request.query)}:{request.top_k}:{request.query_type.value}"
        cached_response = await cache.get(cache_key)
        
        if cached_response and not settings.DEBUG:
            logger.info(f"Cache hit for query [{query_id}]")
            increment_counter("cache_hits_total")
            return QueryResponse.parse_obj(cached_response)
        
        # Get vector store
        vector_store = await get_vector_store()
        
        # Stage 1: Document Retrieval
        retrieval_start = datetime.utcnow()
        logger.debug(f"Starting retrieval for query [{query_id}]")
        
        retrieved_documents = await retrieval_engine.retrieve(
            query=request.query,
            top_k=request.top_k * 2,  # Retrieve more for reranking
            query_type=request.query_type,
            metadata_filters=request.metadata_filters,
            vector_store=vector_store
        )
        
        retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds()
        logger.debug(f"Retrieved {len(retrieved_documents)} documents in {retrieval_time:.3f}s")
        
        # Stage 2: Reranking (if enabled)
        if request.enable_reranking and len(retrieved_documents) > request.top_k:
            rerank_start = datetime.utcnow()
            retrieved_documents = await retrieval_engine.rerank_documents(
                query=request.query,
                documents=retrieved_documents,
                top_k=request.top_k
            )
            rerank_time = (datetime.utcnow() - rerank_start).total_seconds()
            logger.debug(f"Reranked documents in {rerank_time:.3f}s")
        else:
            retrieved_documents = retrieved_documents[:request.top_k]
            rerank_time = 0.0
        
        # Filter by confidence threshold
        if request.confidence_threshold > 0:
            retrieved_documents = [
                doc for doc in retrieved_documents 
                if doc.relevance_score >= request.confidence_threshold
            ]
        
        # Stage 3: Answer Generation
        generation_start = datetime.utcnow()
        logger.debug(f"Starting generation for query [{query_id}]")
        
        generation_result = await generation_service.generate_answer(
            query=request.query,
            retrieved_documents=retrieved_documents,
            stream=request.stream_response
        )
        
        generation_time = (datetime.utcnow() - generation_start).total_seconds()
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Stage 4: Response Assembly
        retrieval_metrics = {
            "documents_retrieved": len(retrieved_documents),
            "avg_relevance_score": sum(doc.relevance_score for doc in retrieved_documents) / len(retrieved_documents) if retrieved_documents else 0.0,
            "retrieval_time_ms": retrieval_time * 1000,
            "reranking_time_ms": rerank_time * 1000,
            "reranking_applied": request.enable_reranking
        }
        
        generation_metrics = {
            "generation_time_ms": generation_time * 1000,
            "response_length": len(generation_result["answer"]),
            "sources_integrated": len(retrieved_documents),
            "confidence_score": generation_result.get("confidence_score", 0.0)
        }
        
        # Create response
        response = QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=generation_result["answer"],
            retrieved_documents=retrieved_documents,
            response_time_ms=total_time * 1000,
            confidence_score=generation_result.get("confidence_score"),
            sources_used=generation_result.get("sources_used", []),
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            metadata={
                "query_type": request.query_type.value,
                "enable_reranking": request.enable_reranking,
                "cache_hit": False
            }
        )
        
        # Cache the response (background task)
        if not settings.DEBUG:
            background_tasks.add_task(
                cache_response, cache, cache_key, response.dict()
            )
        
        # Log metrics (background task)
        background_tasks.add_task(
            log_query_metrics,
            query_id,
            request.query,
            response,
            retrieval_metrics,
            generation_metrics
        )
        
        logger.info(f"Query [{query_id}] processed successfully in {total_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed [{query_id}]: {e}", exc_info=True)
        increment_counter("query_errors_total", {"error_type": type(e).__name__})
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": str(e),
                "query_id": str(query_id),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.post("/stream")
async def stream_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """
    🌊 **Streaming Query Processing**
    
    Provides real-time streaming responses for better user experience.
    Streams answer generation while maintaining full RAG pipeline capabilities.
    """
    query_id = uuid4()
    
    async def generate_stream():
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting streaming query [{query_id}]: {request.query[:100]}...")
            
            # Retrieval phase (same as regular query)
            vector_store = await get_vector_store()
            retrieved_documents = await retrieval_engine.retrieve(
                query=request.query,
                top_k=request.top_k,
                query_type=request.query_type,
                vector_store=vector_store
            )
            
            # Stream generation
            accumulated_answer = ""
            async for chunk in generation_service.generate_streaming_answer(
                query=request.query,
                retrieved_documents=retrieved_documents
            ):
                accumulated_answer += chunk
                
                chunk_response = StreamingQueryResponse(
                    query_id=query_id,
                    chunk=chunk,
                    is_final=False
                )
                
                yield f"data: {chunk_response.json()}\n\n"
            
            # Send final chunk with sources
            final_chunk = StreamingQueryResponse(
                query_id=query_id,
                chunk="",
                is_final=True,
                sources=retrieved_documents
            )
            
            yield f"data: {final_chunk.json()}\n\n"
            
            # Log metrics
            total_time = (datetime.utcnow() - start_time).total_seconds()
            background_tasks.add_task(
                log_streaming_metrics,
                query_id,
                request.query,
                accumulated_answer,
                total_time
            )
            
        except Exception as e:
            logger.error(f"Streaming query failed [{query_id}]: {e}", exc_info=True)
            error_response = StreamingQueryResponse(
                query_id=query_id,
                chunk=f"Error: {str(e)}",
                is_final=True
            )
            yield f"data: {error_response.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/similar")
async def find_similar_content(
    query: str = Query(..., description="Query text", min_length=1, max_length=500),
    top_k: int = Query(5, description="Number of similar documents", ge=1, le=20),
    threshold: float = Query(0.5, description="Similarity threshold", ge=0.0, le=1.0)
) -> List[RetrievedDocument]:
    """
    🔎 **Find Similar Content**
    
    Retrieves similar documents without generating an answer.
    Useful for content discovery and exploration.
    """
    try:
        logger.info(f"Finding similar content for: {query[:100]}...")
        
        vector_store = await get_vector_store()
        similar_documents = await retrieval_engine.retrieve(
            query=query,
            top_k=top_k,
            query_type=QueryType.SEMANTIC,
            vector_store=vector_store
        )
        
        # Filter by threshold
        filtered_documents = [
            doc for doc in similar_documents 
            if doc.relevance_score >= threshold
        ]
        
        logger.info(f"Found {len(filtered_documents)} similar documents")
        return filtered_documents
        
    except Exception as e:
        logger.error(f"Similar content search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Similar content search failed: {str(e)}"
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
) -> FeedbackResponse:
    """
    📝 **Submit User Feedback**
    
    Collects user feedback for query responses to improve system performance.
    Feedback is used for continuous learning and quality assessment.
    """
    try:
        logger.info(f"Receiving feedback for query: {feedback.query_id}")
        
        feedback_response = FeedbackResponse(
            query_id=feedback.query_id,
            feedback_type=feedback.feedback_type,
            rating=feedback.rating,
            comment=feedback.comment,
            helpful=feedback.helpful,
            accurate=feedback.accurate,
            complete=feedback.complete
        )
        
        # Process feedback asynchronously
        background_tasks.add_task(
            process_feedback,
            feedback_response
        )
        
        increment_counter("feedback_submissions_total", 
                         {"feedback_type": feedback.feedback_type.value})
        
        return feedback_response
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feedback submission failed: {str(e)}"
        )


@router.get("/suggestions")
async def get_query_suggestions(
    partial_query: str = Query(..., description="Partial query text", min_length=1),
    limit: int = Query(5, description="Number of suggestions", ge=1, le=10)
) -> List[str]:
    """
    💡 **Get Query Suggestions**
    
    Provides intelligent query suggestions based on partial input
    and historical query patterns.
    """
    try:
        suggestions = await retrieval_engine.get_query_suggestions(
            partial_query=partial_query,
            limit=limit
        )
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Query suggestions failed: {e}", exc_info=True)
        return []  # Return empty list on error to avoid breaking UX


@router.post("/explain")
async def explain_retrieval(
    request: QueryRequest
) -> dict:
    """
    🔍 **Explain Retrieval Process**
    
    Provides detailed explanation of why specific documents were retrieved
    for a given query. Useful for debugging and transparency.
    """
    try:
        vector_store = await get_vector_store()
        retrieved_documents = await retrieval_engine.retrieve(
            query=request.query,
            top_k=request.top_k,
            query_type=request.query_type,
            vector_store=vector_store
        )
        
        explanation = await retrieval_engine.explain_retrieval(
            query=request.query,
            documents=retrieved_documents
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Retrieval explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval explanation failed: {str(e)}"
        )


# Background task functions
async def cache_response(cache, cache_key: str, response_dict: dict):
    """Cache query response"""
    try:
        await cache.set(cache_key, response_dict, ttl=settings.CACHE_TTL)
        logger.debug(f"Cached response for key: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to cache response: {e}")


async def log_query_metrics(
    query_id: uuid4,
    query: str,
    response: QueryResponse,
    retrieval_metrics: dict,
    generation_metrics: dict
):
    """Log comprehensive query metrics"""
    try:
        query_event = QueryEvent(
            query_id=query_id,
            query=query,
            answer=response.answer,
            retrieved_docs=response.retrieved_documents,
            response_time=response.response_time_ms / 1000,
            retrieval_time=retrieval_metrics["retrieval_time_ms"] / 1000,
            generation_time=generation_metrics["generation_time_ms"] / 1000,
            confidence_score=response.confidence_score,
            feedback_score=None,
            timestamp=response.timestamp
        )
        
        await quality_monitor.record_query_event(query_event)
        logger.debug(f"Logged metrics for query: {query_id}")
        
    except Exception as e:
        logger.error(f"Failed to log query metrics: {e}")


async def log_streaming_metrics(
    query_id: uuid4,
    query: str,
    answer: str,
    response_time: float
):
    """Log metrics for streaming queries"""
    try:
        # Log streaming-specific metrics
        logger.info(f"Streaming query [{query_id}] completed in {response_time:.3f}s")
        increment_counter("streaming_queries_total")
        
    except Exception as e:
        logger.error(f"Failed to log streaming metrics: {e}")


async def process_feedback(feedback: FeedbackResponse):
    """Process user feedback for quality improvement"""
    try:
        await quality_monitor.record_feedback(feedback)
        logger.info(f"Processed feedback for query: {feedback.query_id}")
        
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
