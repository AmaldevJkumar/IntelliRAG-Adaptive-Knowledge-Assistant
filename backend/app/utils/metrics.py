"""
Metrics and monitoring utilities
Prometheus integration and custom metrics tracking
"""

import logging
import time
import functools
from typing import Dict, Any, List
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from app.config import get_settings

logger = logging.getLogger(__name__)

# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    # Counters
    queries_total = Counter('rag_queries_total', 'Total number of queries', ['query_type', 'status'])
    documents_processed_total = Counter('rag_documents_processed_total', 'Total documents processed', ['status', 'file_type'])
    cache_hits_total = Counter('rag_cache_hits_total', 'Cache hits', ['backend'])
    cache_misses_total = Counter('rag_cache_misses_total', 'Cache misses')
    
    # Histograms
    query_duration_seconds = Histogram('rag_query_duration_seconds', 'Query processing time')
    retrieval_duration_seconds = Histogram('rag_retrieval_duration_seconds', 'Document retrieval time')
    generation_duration_seconds = Histogram('rag_generation_duration_seconds', 'Answer generation time')
    document_processing_duration_seconds = Histogram('rag_document_processing_duration_seconds', 'Document processing time')
    
    # Gauges
    active_connections = Gauge('rag_active_connections', 'Active connections')
    vector_store_documents = Gauge('rag_vector_store_documents', 'Documents in vector store')
    cache_size = Gauge('rag_cache_size', 'Cache size')
    
    # Info
    system_info = Info('rag_system_info', 'System information')

else:
    logger.warning("Prometheus client not available, using mock metrics")

# In-memory metrics for fallback
_metrics_storage = {
    'counters': {},
    'histograms': {},
    'gauges': {},
    'last_reset': datetime.utcnow()
}


def increment_counter(name: str, labels: Dict[str, str] = None, value: float = 1):
    """Increment a counter metric"""
    try:
        if PROMETHEUS_AVAILABLE:
            if name == 'queries_total':
                queries_total.labels(**(labels or {})).inc(value)
            elif name == 'documents_processed_total':
                documents_processed_total.labels(**(labels or {})).inc(value)
            elif name == 'cache_hits_total':
                cache_hits_total.labels(**(labels or {})).inc(value)
            elif name == 'cache_misses_total':
                cache_misses_total.inc(value)
        else:
            # Fallback to in-memory storage
            key = f"{name}:{labels}" if labels else name
            _metrics_storage['counters'][key] = _metrics_storage['counters'].get(key, 0) + value
            
    except Exception as e:
        logger.error(f"Failed to increment counter {name}: {e}")


def observe_histogram(name: str, value: float, labels: Dict[str, str] = None):
    """Observe a histogram metric"""
    try:
        if PROMETHEUS_AVAILABLE:
            if name == 'query_duration_seconds':
                query_duration_seconds.observe(value)
            elif name == 'retrieval_duration_seconds':
                retrieval_duration_seconds.observe(value)
            elif name == 'generation_duration_seconds':
                generation_duration_seconds.observe(value)
            elif name == 'document_processing_duration_seconds':
                document_processing_duration_seconds.observe(value)
        else:
            # Fallback to in-memory storage
            key = f"{name}:{labels}" if labels else name
            if key not in _metrics_storage['histograms']:
                _metrics_storage['histograms'][key] = []
            _metrics_storage['histograms'][key].append(value)
            
            # Keep only last 1000 observations
            if len(_metrics_storage['histograms'][key]) > 1000:
                _metrics_storage['histograms'][key] = _metrics_storage['histograms'][key][-1000:]
                
    except Exception as e:
        logger.error(f"Failed to observe histogram {name}: {e}")


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Set a gauge metric"""
    try:
        if PROMETHEUS_AVAILABLE:
            if name == 'active_connections':
                active_connections.set(value)
            elif name == 'vector_store_documents':
                vector_store_documents.set(value)
            elif name == 'cache_size':
                cache_size.set(value)
        else:
            # Fallback to in-memory storage
            key = f"{name}:{labels}" if labels else name
            _metrics_storage['gauges'][key] = value
            
    except Exception as e:
        logger.error(f"Failed to set gauge {name}: {e}")


def track_query_time(func):
    """Decorator to track query processing time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            observe_histogram('query_duration_seconds', processing_time)
            increment_counter('queries_total', {'status': 'success'})
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            observe_histogram('query_duration_seconds', processing_time)
            increment_counter('queries_total', {'status': 'error'})
            raise
            
    return wrapper


def track_retrieval_time(func):
    """Decorator to track retrieval time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            retrieval_time = time.time() - start_time
            
            observe_histogram('retrieval_duration_seconds', retrieval_time)
            increment_counter('retrieval_requests_total', {'status': 'success'})
            
            return result
            
        except Exception as e:
            retrieval_time = time.time() - start_time
            observe_histogram('retrieval_duration_seconds', retrieval_time)
            increment_counter('retrieval_requests_total', {'status': 'error'})
            raise
            
    return wrapper


def track_generation_time(func):
    """Decorator to track generation time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            generation_time = time.time() - start_time
            
            observe_histogram('generation_duration_seconds', generation_time)
            increment_counter('generation_requests_total', {'status': 'success'})
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            observe_histogram('generation_duration_seconds', generation_time)
            increment_counter('generation_requests_total', {'status': 'error'})
            raise
            
    return wrapper


def track_processing_time(func):
    """Decorator to track document processing time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            observe_histogram('document_processing_duration_seconds', processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            observe_histogram('document_processing_duration_seconds', processing_time)
            raise
            
    return wrapper


def track_vector_operation(func):
    """Decorator to track vector database operations"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            operation_time = time.time() - start_time
            
            increment_counter('vector_operations_total', {'status': 'success'})
            observe_histogram('vector_operation_duration_seconds', operation_time)
            
            return result
            
        except Exception as e:
            operation_time = time.time() - start_time
            increment_counter('vector_operations_total', {'status': 'error'})
            observe_histogram('vector_operation_duration_seconds', operation_time)
            raise
            
    return wrapper


async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    try:
        if PROMETHEUS_AVAILABLE:
            # In a real implementation, this would query Prometheus
            # For now, return mock data
            return {
                "uptime_hours": 247.5,
                "total_queries": 47892,
                "avg_response_time_ms": 1247.8,
                "avg_confidence_score": 0.89,
                "cache_hit_rate": 0.74,
                "active_documents": 15847,
                "vector_db_status": "optimal",
                "llm_service_status": "available",
                "memory_usage_mb": 1024.5,
                "cpu_usage_percent": 45.2
            }
        else:
            # Use in-memory metrics
            return {
                "counters": dict(_metrics_storage['counters']),
                "gauges": dict(_metrics_storage['gauges']),
                "histogram_counts": {
                    k: len(v) for k, v in _metrics_storage['histograms'].items()
                },
                "last_reset": _metrics_storage['last_reset'].isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {}


async def get_prometheus_metrics() -> str:
    """Get metrics in Prometheus format"""
    try:
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest().decode('utf-8')
        else:
            # Generate simple text format from in-memory metrics
            lines = ["# Fallback metrics (Prometheus not available)"]
            
            for name, value in _metrics_storage['counters'].items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            
            for name, value in _metrics_storage['gauges'].items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            
            return '\n'.join(lines)
            
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        return "# Error retrieving metrics"


async def export_metrics(format_type: str = "json", time_range: str = "24h") -> Dict[str, Any]:
    """Export metrics in various formats"""
    try:
        base_metrics = await get_system_metrics()
        
        if format_type == "json":
            return {
                "format": "json",
                "timestamp": datetime.utcnow().isoformat(),
                "time_range": time_range,
                "metrics": base_metrics
            }
        elif format_type == "csv":
            # Convert to CSV format
            csv_data = "metric_name,value,timestamp\n"
            timestamp = datetime.utcnow().isoformat()
            
            for key, value in base_metrics.items():
                if isinstance(value, (int, float)):
                    csv_data += f"{key},{value},{timestamp}\n"
            
            return {"format": "csv", "data": csv_data}
        elif format_type == "prometheus":
            return {
                "format": "prometheus", 
                "data": await get_prometheus_metrics()
            }
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        return {"error": str(e)}


def reset_metrics():
    """Reset in-memory metrics (for testing)"""
    global _metrics_storage
    _metrics_storage = {
        'counters': {},
        'histograms': {},
        'gauges': {},
        'last_reset': datetime.utcnow()
    }
    logger.info("Metrics reset")


def initialize_metrics():
    """Initialize metrics system"""
    try:
        if PROMETHEUS_AVAILABLE:
            # Set system info
            settings = get_settings()
            system_info.info({
                'version': settings.APP_VERSION,
                'environment': 'production' if settings.is_production else 'development',
                'vector_db': settings.VECTOR_DB_TYPE,
                'llm_provider': settings.LLM_PROVIDER
            })
            
            logger.info("Prometheus metrics initialized")
        else:
            logger.info("Using fallback in-memory metrics")
            
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")


# Custom metrics context manager
class MetricsContext:
    """Context manager for tracking operation metrics"""
    
    def __init__(self, operation_name: str, labels: Dict[str, str] = None):
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        increment_counter(f'{self.operation_name}_started_total', self.labels)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            increment_counter(f'{self.operation_name}_completed_total', self.labels)
            observe_histogram(f'{self.operation_name}_duration_seconds', duration)
        else:
            increment_counter(f'{self.operation_name}_failed_total', self.labels)
            observe_histogram(f'{self.operation_name}_duration_seconds', duration)
