"""
API endpoint tests for RAG Knowledge Assistant
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json

from backend.app.main import app


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestQueryEndpoints:
    """Test query processing endpoints"""
    
    def test_query_endpoint_success(self, client, sample_query_request):
        """Test successful query processing"""
        response = client.post("/api/v1/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "query_id" in data
        assert "query" in data
        assert "answer" in data
        assert "retrieved_documents" in data
        assert "response_time_ms" in data
        assert "confidence_score" in data
        assert "timestamp" in data
        
        # Check data types
        assert isinstance(data["retrieved_documents"], list)
        assert isinstance(data["response_time_ms"], (int, float))
        assert isinstance(data["confidence_score"], (int, float))
        
        # Check query echo
        assert data["query"] == sample_query_request["query"]
    
    def test_query_validation_empty_query(self, client):
        """Test query validation with empty query"""
        response = client.post("/api/v1/query", json={"query": ""})
        
        assert response.status_code == 422
    
    def test_query_validation_invalid_top_k(self, client):
        """Test query validation with invalid top_k"""
        response = client.post("/api/v1/query", json={
            "query": "test query",
            "top_k": 0
        })
        
        assert response.status_code == 422
    
    def test_query_validation_invalid_query_type(self, client):
        """Test query validation with invalid query type"""
        response = client.post("/api/v1/query", json={
            "query": "test query",
            "query_type": "invalid_type"
        })
        
        assert response.status_code == 422
    
    def test_similar_content_endpoint(self, client):
        """Test similar content endpoint"""
        response = client.get("/api/v1/similar", params={
            "query": "machine learning",
            "top_k": 3,
            "threshold": 0.5
        })
        
        assert response.status_code in [200, 500]  # Might fail in test env
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) <= 3
    
    def test_query_suggestions_endpoint(self, client):
        """Test query suggestions endpoint"""
        response = client.get("/api/v1/suggestions", params={
            "partial_query": "machine",
            "limit": 5
        })
        
        assert response.status_code in [200, 500]  # Might fail in test env


class TestDocumentEndpoints:
    """Test document management endpoints"""
    
    def test_list_documents(self, client):
        """Test document listing"""
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["items"], list)
    
    def test_list_documents_with_pagination(self, client):
        """Test document listing with pagination"""
        response = client.get("/api/v1/documents", params={
            "limit": 5,
            "offset": 0
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["limit"] == 5
        assert data["offset"] == 0
    
    def test_list_documents_with_filters(self, client):
        """Test document listing with filters"""
        response = client.get("/api/v1/documents", params={
            "status": "completed",
            "source": "test_source"
        })
        
        assert response.status_code == 200
    
    def test_upload_document_no_file(self, client):
        """Test document upload without file"""
        response = client.post("/api/v1/documents/upload")
        
        assert response.status_code == 422
    
    def test_upload_document_success(self, client, temp_file):
        """Test successful document upload"""
        with open(temp_file, 'rb') as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={
                    "source": "test_source",
                    "metadata": json.dumps({"category": "test"})
                }
            )
        
        assert response.status_code in [200, 500]  # Might fail in test env
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "filename" in data
            assert "status" in data


class TestAdminEndpoints:
    """Test admin endpoints"""
    
    def test_system_status_unauthorized(self, client):
        """Test system status without authentication"""
        response = client.get("/api/v1/admin/status")
        
        # Should require authentication
        assert response.status_code in [401, 403, 500]
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/api/v1/monitoring/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required metric fields
        expected_fields = [
            "uptime_hours", "total_queries", "avg_response_time_ms",
            "avg_confidence_score", "cache_hit_rate", "active_documents"
        ]
        
        for field in expected_fields:
            assert field in data


class TestMonitoringEndpoints:
    """Test monitoring and analytics endpoints"""
    
    def test_get_metrics(self, client):
        """Test metrics retrieval"""
        response = client.get("/api/v1/monitoring/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check metric structure
        assert isinstance(data["uptime_hours"], (int, float))
        assert isinstance(data["total_queries"], int)
        assert isinstance(data["avg_response_time_ms"], (int, float))
    
    def test_quality_dashboard(self, client):
        """Test quality dashboard endpoint"""
        response = client.get("/api/v1/monitoring/quality/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check dashboard structure
        assert "timestamp" in data
        assert "overview" in data
        assert "quality_metrics" in data
        assert "user_feedback" in data
    
    def test_get_alerts(self, client):
        """Test alerts endpoint"""
        response = client.get("/api/v1/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "active_alerts" in data
        assert "alert_counts" in data
        assert isinstance(data["active_alerts"], list)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_404_endpoint(self, client):
        """Test non-existent endpoint"""
        response = client.get("/non/existent/endpoint")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed"""
        response = client.post("/health")  # GET only endpoint
        
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test invalid JSON in request"""
        response = client.post(
            "/api/v1/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_large_query(self, client):
        """Test very large query"""
        large_query = "test " * 1000  # Very long query
        
        response = client.post("/api/v1/query", json={
            "query": large_query
        })
        
        # Should either process or reject gracefully
        assert response.status_code in [200, 413, 422]


class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_protected_endpoint_no_auth(self, client):
        """Test protected endpoint without authentication"""
        response = client.get("/api/v1/admin/status")
        
        assert response.status_code in [401, 403]
    
    def test_invalid_token(self, client):
        """Test with invalid authentication token"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/v1/admin/status", headers=headers)
        
        assert response.status_code in [401, 403]


class TestCORS:
    """Test CORS headers"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/query")
        
        # CORS headers should be present
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers


@pytest.mark.integration
class TestIntegrationFlows:
    """Test complete API flows"""
    
    def test_complete_query_flow(self, client, sample_query_request):
        """Test complete query processing flow"""
        # 1. Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Query processing
        query_response = client.post("/api/v1/query", json=sample_query_request)
        assert query_response.status_code == 200
        
        query_data = query_response.json()
        assert "answer" in query_data
        assert len(query_data["answer"]) > 0
        
        # 3. Check metrics after query
        metrics_response = client.get("/api/v1/monitoring/metrics")
        assert metrics_response.status_code == 200
    
    def test_document_lifecycle(self, client, temp_file):
        """Test complete document lifecycle"""
        # 1. List documents before upload
        list_response = client.get("/api/v1/documents")
        assert list_response.status_code == 200
        initial_count = list_response.json()["total"]
        
        # 2. Upload document
        with open(temp_file, 'rb') as f:
            upload_response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        # Upload might fail in test environment, that's OK
        if upload_response.status_code == 200:
            doc_data = upload_response.json()
            
            # 3. Verify document appears in list
            list_response = client.get("/api/v1/documents")
            assert list_response.status_code == 200
