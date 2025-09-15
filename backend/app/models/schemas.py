"""
Pydantic models for request/response schemas
Comprehensive data validation and API documentation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator


class QueryType(str, Enum):
    """Supported query types"""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FeedbackType(str, Enum):
    """User feedback types"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    COMMENT = "comment"


# Request Models
class QueryRequest(BaseModel):
    """Advanced query request with comprehensive parameters"""
    query: str = Field(..., description="User query text", min_length=1, max_length=2000)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)
    query_type: QueryType = Field(QueryType.HYBRID, description="Search strategy")
    include_sources: bool = Field(True, description="Include source document references")
    confidence_threshold: float = Field(0.7, description="Minimum confidence score", ge=0.0, le=1.0)
    enable_reranking: bool = Field(True, description="Apply cross-encoder reranking")
    stream_response: bool = Field(False, description="Enable streaming response")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()


class DocumentUploadRequest(BaseModel):
    """Document upload request"""
    filename: str = Field(..., description="Document filename")
    source: Optional[str] = Field(None, description="Document source")
    department: Optional[str] = Field(None, description="Department/category")
    tags: List[str] = Field([], description="Document tags")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")


class FeedbackRequest(BaseModel):
    """User feedback request"""
    query_id: UUID = Field(..., description="Query ID to provide feedback for")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    rating: Optional[int] = Field(None, description="Rating (1-5)", ge=1, le=5)
    comment: Optional[str] = Field(None, description="Feedback comment", max_length=1000)
    helpful: Optional[bool] = Field(None, description="Was the response helpful")
    accurate: Optional[bool] = Field(None, description="Was the response accurate")
    complete: Optional[bool] = Field(None, description="Was the response complete")
    
    @validator('rating')
    def validate_rating(cls, v, values):
        if values.get('feedback_type') == FeedbackType.RATING and v is None:
            raise ValueError('Rating is required for rating feedback type')
        return v


# Response Models
class RetrievedDocument(BaseModel):
    """Retrieved document with comprehensive metadata"""
    document_id: str = Field(..., description="Unique document identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Document content/excerpt")
    relevance_score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    chunk_index: int = Field(..., description="Chunk position in document", ge=0)
    filename: str = Field(..., description="Original filename")
    source: Optional[str] = Field(None, description="Document source")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")


class QueryResponse(BaseModel):
    """Comprehensive query response"""
    query_id: UUID = Field(default_factory=uuid4, description="Unique query identifier")
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated response")
    retrieved_documents: List[RetrievedDocument] = Field([], description="Source documents")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    confidence_score: Optional[float] = Field(None, description="Response confidence", ge=0.0, le=1.0)
    sources_used: List[str] = Field([], description="List of source filenames used")
    retrieval_metrics: Dict[str, float] = Field({}, description="Retrieval performance metrics")
    generation_metrics: Dict[str, float] = Field({}, description="Generation performance metrics")
    metadata: Dict[str, Any] = Field({}, description="Additional response metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class StreamingQueryResponse(BaseModel):
    """Streaming query response chunk"""
    query_id: UUID = Field(..., description="Query identifier")
    chunk: str = Field(..., description="Response chunk")
    is_final: bool = Field(False, description="Is this the final chunk")
    sources: List[RetrievedDocument] = Field([], description="Source documents (in final chunk)")


class DocumentResponse(BaseModel):
    """Document information response"""
    id: UUID = Field(..., description="Document unique identifier")
    filename: str = Field(..., description="Document filename")
    status: DocumentStatus = Field(..., description="Processing status")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    source: Optional[str] = Field(None, description="Document source")
    department: Optional[str] = Field(None, description="Department/category")
    tags: List[str] = Field([], description="Document tags")
    chunk_count: Optional[int] = Field(None, description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")


class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    feedback_id: UUID = Field(default_factory=uuid4, description="Feedback identifier")
    query_id: UUID = Field(..., description="Associated query ID")
    feedback_type: FeedbackType = Field(..., description="Feedback type")
    rating: Optional[int] = Field(None, description="Rating value")
    comment: Optional[str] = Field(None, description="Feedback comment")
    helpful: Optional[bool] = Field(None, description="Helpfulness rating")
    accurate: Optional[bool] = Field(None, description="Accuracy rating")
    complete: Optional[bool] = Field(None, description="Completeness rating")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Submission timestamp")


# System Models
class SystemInfo(BaseModel):
    """System information and capabilities"""
    name: str = Field(..., description="System name")
    version: str = Field(..., description="System version")
    description: str = Field(..., description="System description")
    status: str = Field(..., description="System status")
    capabilities: Dict[str, List[str]] = Field(..., description="System capabilities")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    tech_stack: Dict[str, Any] = Field(..., description="Technology stack information")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    components: Dict[str, Dict[str, Any]] = Field({}, description="Component health status")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class SystemMetrics(BaseModel):
    """System performance metrics"""
    uptime_hours: float = Field(..., description="System uptime in hours")
    total_queries: int = Field(..., description="Total queries processed")
    avg_response_time_ms: float = Field(..., description="Average response time")
    avg_confidence_score: float = Field(..., description="Average confidence score")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    active_documents: int = Field(..., description="Number of active documents")
    vector_db_status: str = Field(..., description="Vector database status")
    llm_service_status: str = Field(..., description="LLM service status")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")


# Evaluation Models
class EvaluationRequest(BaseModel):
    """Request for system evaluation"""
    query: str = Field(..., description="Test query")
    expected_answer: Optional[str] = Field(None, description="Expected answer")
    expected_sources: List[str] = Field([], description="Expected source documents")
    evaluation_metrics: List[str] = Field([], description="Metrics to evaluate")


class EvaluationResult(BaseModel):
    """Evaluation result for a single query"""
    query: str = Field(..., description="Test query")
    generated_answer: str = Field(..., description="Generated answer")
    retrieved_documents: List[RetrievedDocument] = Field([], description="Retrieved documents")
    faithfulness_score: Optional[float] = Field(None, description="Faithfulness score")
    answer_relevancy_score: Optional[float] = Field(None, description="Answer relevancy score")
    context_precision_score: Optional[float] = Field(None, description="Context precision score")
    context_recall_score: Optional[float] = Field(None, description="Context recall score")
    overall_score: Optional[float] = Field(None, description="Overall evaluation score")
    evaluation_time_ms: float = Field(..., description="Evaluation time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request"""
    queries: List[EvaluationRequest] = Field(..., description="List of queries to evaluate")
    run_name: str = Field(..., description="Evaluation run name")
    description: Optional[str] = Field(None, description="Run description")


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation response"""
    run_id: UUID = Field(default_factory=uuid4, description="Evaluation run ID")
    run_name: str = Field(..., description="Evaluation run name")
    total_queries: int = Field(..., description="Total number of queries")
    completed_queries: int = Field(..., description="Number of completed queries")
    avg_overall_score: float = Field(..., description="Average overall score")
    results: List[EvaluationResult] = Field([], description="Individual evaluation results")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    limit: int = Field(20, description="Number of items per page", ge=1, le=100)
    offset: int = Field(0, description="Number of items to skip", ge=0)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Items skipped")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")


# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="Invalid value provided")
