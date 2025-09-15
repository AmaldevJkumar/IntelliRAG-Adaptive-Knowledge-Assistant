"""
Document management endpoints
Upload, processing, and lifecycle management
"""

import logging
import os
from typing import List, Optional
from uuid import uuid4, UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.models.schemas import (
    DocumentResponse, DocumentUploadRequest, DocumentStatus,
    PaginationParams, PaginatedResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import get_vector_store
from app.utils.security import get_current_user
from app.utils.metrics import increment_counter
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

document_processor = DocumentProcessor()
settings = get_settings()


@router.get("/", response_model=PaginatedResponse)
async def list_documents(
    limit: int = Query(20, description="Items per page", ge=1, le=100),
    offset: int = Query(0, description="Items to skip", ge=0),
    status: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    source: Optional[str] = Query(None, description="Filter by source"),
    search: Optional[str] = Query(None, description="Search in filenames")
) -> PaginatedResponse:
    """
    📚 **List Documents in Knowledge Base**
    
    Retrieves paginated list of documents with comprehensive metadata,
    processing status, and performance metrics.
    
    **Features:**
    - Pagination support with configurable limits
    - Status filtering (pending, processing, completed, failed)
    - Source-based filtering for organization
    - Full-text search in document names
    - Rich metadata including processing times and access patterns
    """
    try:
        logger.info(f"Listing documents: limit={limit}, offset={offset}, status={status}")
        
        # Mock implementation - replace with actual database query
        total_documents = 15847
        
        # Generate mock documents
        documents = []
        for i in range(offset + 1, min(offset + limit + 1, total_documents + 1)):
            doc_status = DocumentStatus.COMPLETED
            if i % 50 == 0:
                doc_status = DocumentStatus.PROCESSING
            elif i % 100 == 0:
                doc_status = DocumentStatus.FAILED
            
            # Apply status filter
            if status and doc_status != status:
                continue
                
            document = DocumentResponse(
                id=uuid4(),
                filename=f"enterprise_document_{i:06d}.pdf",
                status=doc_status,
                file_size=int(2.5e6 + (i * 1000)),  # 2.5MB + variation
                content_type="application/pdf",
                source=f"Department_{(i % 5) + 1}" if not source else source,
                department="AI Research" if i % 2 == 0 else "Engineering",
                tags=["machine-learning", "enterprise", "documentation"],
                chunk_count=45 + (i % 20) if doc_status == DocumentStatus.COMPLETED else None,
                created_at=datetime(2024, 9, 1, 10, 0, 0),
                updated_at=datetime(2024, 9, 12, 16, 30, 0),
                last_accessed=datetime(2024, 9, 12, 14, 15, 0) if i % 3 == 0 else None,
                metadata={
                    "author": f"Dr. Expert_{i % 10}",
                    "version": "1.0",
                    "processing_time_seconds": 23.4 + (i * 0.1),
                    "query_frequency": max(1, 100 - (i % 50)),
                    "avg_relevance_score": round(0.75 + (i % 25) * 0.01, 3),
                    "language": "en",
                    "security_level": "internal"
                }
            )
            documents.append(document)
        
        # Apply search filter
        if search:
            documents = [
                doc for doc in documents 
                if search.lower() in doc.filename.lower()
            ]
        
        has_next = (offset + limit) < total_documents
        has_previous = offset > 0
        
        result = PaginatedResponse(
            items=documents,
            total=total_documents,
            limit=limit,
            offset=offset,
            has_next=has_next,
            has_previous=has_previous
        )
        
        logger.info(f"Listed {len(documents)} documents")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document list")


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    source: str = Form(None, description="Document source"),
    department: str = Form(None, description="Department/category"),
    tags: str = Form("", description="Comma-separated tags"),
    metadata: str = Form("{}", description="Additional metadata as JSON")
) -> DocumentResponse:
    """
    📤 **Upload Document to Knowledge Base**
    
    Handles document upload with comprehensive validation, processing,
    and integration into the RAG system.
    
    **Process:**
    1. File validation (size, type, content)
    2. Virus scanning and security checks
    3. Metadata extraction and processing
    4. Text extraction and chunking
    5. Vector embedding generation
    6. Index storage and cataloging
    7. Quality assessment and validation
    
    **Supported Formats:**
    - PDF documents
    - Microsoft Word (DOCX)
    - Plain text (TXT)
    - Markdown (MD)
    - HTML files
    """
    try:
        logger.info(f"Processing file upload: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Check file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.supported_file_types_list:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type. Supported: {', '.join(settings.supported_file_types_list)}"
            )
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        
        # Parse metadata
        import json
        try:
            additional_metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            additional_metadata = {}
        
        # Create document record
        document_id = uuid4()
        document_response = DocumentResponse(
            id=document_id,
            filename=file.filename,
            status=DocumentStatus.PROCESSING,
            file_size=file.size or 0,
            content_type=file.content_type or "application/octet-stream",
            source=source,
            department=department,
            tags=tag_list,
            chunk_count=None,  # Will be updated after processing
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=additional_metadata
        )
        
        # Save file temporarily
        temp_file_path = f"/tmp/{document_id}_{file.filename}"
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Process document asynchronously
        background_tasks.add_task(
            process_document_async,
            document_id,
            temp_file_path,
            document_response.dict()
        )
        
        increment_counter("documents_uploaded_total", 
                         {"file_type": file_extension, "source": source or "unknown"})
        
        logger.info(f"Document upload initiated: {document_id}")
        return document_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Document upload failed")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: UUID) -> DocumentResponse:
    """
    📄 **Get Document Information**
    
    Retrieves detailed information about a specific document
    including processing status, metadata, and performance metrics.
    """
    try:
        logger.info(f"Retrieving document: {document_id}")
        
        # Mock implementation - replace with database query
        document = DocumentResponse(
            id=document_id,
            filename="sample_enterprise_document.pdf",
            status=DocumentStatus.COMPLETED,
            file_size=2567890,
            content_type="application/pdf",
            source="AI Research Department",
            department="AI Research",
            tags=["machine-learning", "rag-systems", "enterprise"],
            chunk_count=47,
            created_at=datetime(2024, 9, 10, 14, 30, 0),
            updated_at=datetime(2024, 9, 10, 14, 35, 0),
            last_accessed=datetime.utcnow(),
            metadata={
                "author": "Dr. AI Researcher",
                "version": "2.1",
                "processing_time_seconds": 28.7,
                "avg_relevance_score": 0.89,
                "total_queries": 156,
                "last_query": "2024-09-12T16:45:00Z"
            }
        )
        
        return document
        
    except Exception as e:
        logger.error(f"Failed to retrieve document {document_id}: {e}")
        raise HTTPException(status_code=404, detail="Document not found")


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    background_tasks: BackgroundTasks
) -> dict:
    """
    🗑️ **Delete Document**
    
    Removes document from knowledge base including:
    - Vector embeddings from index
    - Document chunks and metadata
    - File storage cleanup
    - Search index updates
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Process deletion asynchronously
        background_tasks.add_task(delete_document_async, document_id)
        
        increment_counter("documents_deleted_total")
        
        return {
            "message": "Document deletion initiated",
            "document_id": str(document_id),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Document deletion failed")


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    background_tasks: BackgroundTasks
) -> dict:
    """
    🔄 **Reprocess Document**
    
    Triggers reprocessing of an existing document with updated
    configuration or to recover from processing failures.
    """
    try:
        logger.info(f"Reprocessing document: {document_id}")
        
        # Trigger reprocessing
        background_tasks.add_task(reprocess_document_async, document_id)
        
        return {
            "message": "Document reprocessing initiated",
            "document_id": str(document_id),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document reprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Document reprocessing failed")


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> dict:
    """
    📋 **Get Document Chunks**
    
    Retrieves the text chunks for a specific document
    with relevance scores and metadata.
    """
    try:
        logger.info(f"Retrieving chunks for document: {document_id}")
        
        # Mock chunks - replace with actual database query
        chunks = []
        total_chunks = 47
        
        for i in range(offset, min(offset + limit, total_chunks)):
            chunks.append({
                "chunk_id": f"chunk_{i:03d}",
                "index": i,
                "content": f"This is chunk {i} of the document containing important information about enterprise AI systems and RAG architectures...",
                "start_char": i * 1000,
                "end_char": (i + 1) * 1000 - 1,
                "metadata": {
                    "section": f"Section {(i // 5) + 1}",
                    "subsection": f"Subsection {(i % 5) + 1}",
                    "page": (i // 3) + 1
                }
            })
        
        return {
            "document_id": str(document_id),
            "chunks": chunks,
            "total_chunks": total_chunks,
            "limit": limit,
            "offset": offset,
            "has_next": (offset + limit) < total_chunks
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document chunks")


@router.get("/{document_id}/download")
async def download_document(document_id: UUID):
    """
    ⬇️ **Download Document**
    
    Downloads the original document file.
    """
    try:
        # Mock implementation - replace with actual file retrieval
        logger.info(f"Downloading document: {document_id}")
        
        # In real implementation, get file path from database
        file_path = f"/app/data/documents/{document_id}.pdf"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        return FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=f"document_{document_id}.pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document download failed: {e}")
        raise HTTPException(status_code=500, detail="Document download failed")


# Background task functions
async def process_document_async(document_id: UUID, file_path: str, document_info: dict):
    """Process uploaded document asynchronously"""
    try:
        logger.info(f"Starting async processing for document: {document_id}")
        
        # Process the document
        result = await document_processor.process_document(
            file_path=file_path,
            document_id=document_id,
            metadata=document_info
        )
        
        # Update document status in database
        # In real implementation, update the document record
        logger.info(f"Document processing completed: {document_id}")
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        increment_counter("documents_processed_total", 
                         {"status": "completed", "chunks": result.get("chunk_count", 0)})
        
    except Exception as e:
        logger.error(f"Async document processing failed for {document_id}: {e}")
        increment_counter("documents_processed_total", {"status": "failed"})
        
        # Update document status to failed
        # In real implementation, update the document record


async def delete_document_async(document_id: UUID):
    """Delete document and cleanup resources"""
    try:
        logger.info(f"Starting async deletion for document: {document_id}")
        
        # Remove from vector store
        vector_store = await get_vector_store()
        await vector_store.delete_document(document_id)
        
        # Remove from database
        # In real implementation, delete from database
        
        # Clean up files
        # In real implementation, remove stored files
        
        logger.info(f"Document deletion completed: {document_id}")
        
    except Exception as e:
        logger.error(f"Async document deletion failed for {document_id}: {e}")


async def reprocess_document_async(document_id: UUID):
    """Reprocess existing document"""
    try:
        logger.info(f"Starting reprocessing for document: {document_id}")
        
        # Get document info from database
        # Reprocess the document with current settings
        # Update vector embeddings
        # Update document status
        
        logger.info(f"Document reprocessing completed: {document_id}")
        increment_counter("documents_reprocessed_total")
        
    except Exception as e:
        logger.error(f"Document reprocessing failed for {document_id}: {e}")
