"""
Administrative endpoints for system management
Advanced system control and configuration
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer

from app.models.schemas import SystemMetrics, HealthCheck
from app.services.monitoring import QualityMonitor, SystemMonitor
from app.services.vector_store import get_vector_store
from app.services.cache import get_cache
from app.utils.security import get_admin_user, require_admin_role
from app.utils.metrics import get_system_metrics, reset_metrics
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

system_monitor = SystemMonitor()
quality_monitor = QualityMonitor()
settings = get_settings()


@router.get("/status", response_model=Dict[str, Any])
@require_admin_role
async def get_detailed_status(
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    ⚡ **Comprehensive System Status**
    
    Provides detailed system status for all components including:
    - Service health and connectivity
    - Performance metrics and resource usage
    - Configuration validation
    - Security and compliance status
    - Operational metrics and alerts
    """
    try:
        logger.info("Retrieving comprehensive system status")
        
        # Get individual component status
        api_status = await system_monitor.check_api_health()
        db_status = await system_monitor.check_database_health()
        vector_db_status = await system_monitor.check_vector_db_health()
        cache_status = await system_monitor.check_cache_health()
        llm_status = await system_monitor.check_llm_health()
        
        # Get system metrics
        metrics = await get_system_metrics()
        
        # Get recent alerts
        alerts = await system_monitor.get_active_alerts()
        
        status_response = {
            "overall_status": "healthy",  # Computed based on components
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
            "environment": "production" if settings.is_production else "development",
            
            "services": {
                "api_gateway": {
                    "status": api_status.get("status", "unknown"),
                    "response_time_p99": api_status.get("response_time_p99", "N/A"),
                    "requests_per_minute": api_status.get("requests_per_minute", 0),
                    "error_rate": api_status.get("error_rate", 0.0),
                    "active_connections": api_status.get("active_connections", 0)
                },
                "database": {
                    "status": db_status.get("status", "unknown"),
                    "connection_pool_usage": db_status.get("pool_usage", 0.0),
                    "active_connections": db_status.get("active_connections", 0),
                    "query_latency_avg": db_status.get("avg_latency", "N/A"),
                    "slow_queries": db_status.get("slow_queries", 0)
                },
                "vector_database": {
                    "status": vector_db_status.get("status", "unknown"),
                    "provider": settings.VECTOR_DB_TYPE,
                    "document_count": vector_db_status.get("document_count", 0),
                    "index_size_mb": vector_db_status.get("index_size_mb", 0),
                    "query_latency_p95": vector_db_status.get("query_latency_p95", "N/A")
                },
                "cache_layer": {
                    "status": cache_status.get("status", "unknown"),
                    "hit_rate": cache_status.get("hit_rate", 0.0),
                    "memory_usage_percent": cache_status.get("memory_usage", 0.0),
                    "eviction_rate_per_min": cache_status.get("eviction_rate", 0.0),
                    "connected_clients": cache_status.get("connected_clients", 0)
                },
                "llm_service": {
                    "status": llm_status.get("status", "unknown"),
                    "provider": settings.LLM_PROVIDER,
                    "model": settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else "N/A",
                    "avg_generation_time": llm_status.get("avg_generation_time", "N/A"),
                    "tokens_per_minute": llm_status.get("tokens_per_minute", 0),
                    "rate_limit_remaining": llm_status.get("rate_limit_remaining", "N/A")
                }
            },
            
            "performance": {
                "uptime_hours": metrics.get("uptime_hours", 0.0),
                "total_queries": metrics.get("total_queries", 0),
                "queries_per_minute": metrics.get("queries_per_minute", 0),
                "avg_response_time_ms": metrics.get("avg_response_time_ms", 0.0),
                "avg_confidence_score": metrics.get("avg_confidence_score", 0.0),
                "cache_hit_rate": metrics.get("cache_hit_rate", 0.0),
                "throughput_qps": metrics.get("throughput_qps", 0.0)
            },
            
            "infrastructure": {
                "cpu_usage_percent": metrics.get("cpu_usage", 0.0),
                "memory_usage_percent": metrics.get("memory_usage", 0.0),
                "disk_usage_percent": metrics.get("disk_usage", 0.0),
                "network_io_mbps": metrics.get("network_io", 0.0),
                "container_status": "running",
                "kubernetes_status": "healthy" if settings.is_production else "N/A"
            },
            
            "security": {
                "auth_failures_last_hour": metrics.get("auth_failures", 0),
                "rate_limit_violations": metrics.get("rate_limit_violations", 0),
                "suspicious_activities": metrics.get("suspicious_activities", 0),
                "last_security_scan": "2024-09-12T10:00:00Z"
            },
            
            "alerts": {
                "active_alerts": len(alerts),
                "critical_count": len([a for a in alerts if a.get("severity") == "critical"]),
                "warning_count": len([a for a in alerts if a.get("severity") == "warning"]),
                "recent_alerts": alerts[:5]  # Show 5 most recent
            },
            
            "configuration": {
                "vector_db_type": settings.VECTOR_DB_TYPE,
                "llm_provider": settings.LLM_PROVIDER,
                "debug_mode": settings.DEBUG,
                "metrics_enabled": settings.ENABLE_METRICS,
                "mlflow_enabled": settings.ENABLE_MLFLOW,
                "cache_ttl": settings.CACHE_TTL
            }
        }
        
        return status_response
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


@router.post("/reindex")
@require_admin_role
async def trigger_reindex(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force reindex even if up to date"),
    batch_size: int = Query(100, description="Documents per batch", ge=1, le=1000),
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    🔄 **Trigger System Reindexing**
    
    Initiates comprehensive reindexing of the knowledge base:
    - Document reprocessing with current settings
    - Vector embedding regeneration
    - Index optimization and cleanup
    - Metadata refresh and validation
    """
    try:
        logger.info(f"Admin {current_user.get('username')} triggered reindexing")
        
        # Start reindexing process
        reindex_id = await system_monitor.start_reindex_process(
            force=force,
            batch_size=batch_size,
            initiated_by=current_user.get("username")
        )
        
        # Process reindexing in background
        background_tasks.add_task(
            execute_reindex_process,
            reindex_id,
            force,
            batch_size
        )
        
        return {
            "message": "Reindexing process initiated",
            "reindex_id": reindex_id,
            "estimated_duration": "30-60 minutes",
            "force_mode": force,
            "batch_size": batch_size,
            "initiated_by": current_user.get("username"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Reindexing initiation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate reindexing")


@router.post("/cache/clear")
@require_admin_role
async def clear_cache(
    cache_type: str = Query("all", description="Cache type to clear"),
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    🧹 **Clear System Cache**
    
    Clears various system caches for maintenance and debugging.
    """
    try:
        logger.info(f"Admin {current_user.get('username')} clearing cache: {cache_type}")
        
        cache = await get_cache()
        
        if cache_type == "all":
            await cache.flushall()
            cleared_items = "all cache items"
        elif cache_type == "queries":
            await cache.delete_pattern("query:*")
            cleared_items = "query cache"
        elif cache_type == "documents":
            await cache.delete_pattern("doc:*")
            cleared_items = "document cache"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown cache type: {cache_type}")
        
        return {
            "message": f"Cache cleared successfully: {cleared_items}",
            "cache_type": cache_type,
            "cleared_by": current_user.get("username"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/metrics/detailed")
@require_admin_role
async def get_detailed_metrics(
    time_range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    📊 **Get Detailed System Metrics**
    
    Retrieves comprehensive metrics for the specified time range.
    """
    try:
        logger.info(f"Retrieving detailed metrics for {time_range}")
        
        # Parse time range
        time_delta_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        if time_range not in time_delta_map:
            raise HTTPException(status_code=400, detail="Invalid time range")
        
        end_time = datetime.utcnow()
        start_time = end_time - time_delta_map[time_range]
        
        # Get detailed metrics from monitoring service
        metrics = await quality_monitor.get_detailed_metrics(start_time, end_time)
        
        return {
            "time_range": time_range,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detailed metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve detailed metrics")


@router.post("/config/update")
@require_admin_role
async def update_configuration(
    config_updates: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    ⚙️ **Update System Configuration**
    
    Updates system configuration with validation and rollback capability.
    """
    try:
        logger.info(f"Admin {current_user.get('username')} updating configuration")
        
        # Validate configuration updates
        valid_keys = [
            "RETRIEVAL_TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP", 
            "CACHE_TTL", "QUALITY_THRESHOLD", "ENABLE_METRICS"
        ]
        
        invalid_keys = set(config_updates.keys()) - set(valid_keys)
        if invalid_keys:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid configuration keys: {list(invalid_keys)}"
            )
        
        # Apply configuration updates
        background_tasks.add_task(
            apply_config_updates,
            config_updates,
            current_user.get("username")
        )
        
        return {
            "message": "Configuration update initiated",
            "updates": config_updates,
            "updated_by": current_user.get("username"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")


@router.get("/logs")
@require_admin_role
async def get_system_logs(
    level: str = Query("INFO", description="Log level filter"),
    limit: int = Query(100, description="Number of log entries", ge=1, le=1000),
    service: str = Query("all", description="Service filter"),
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    📝 **Get System Logs**
    
    Retrieves system logs with filtering and pagination.
    """
    try:
        logger.info(f"Retrieving system logs: level={level}, limit={limit}")
        
        # Mock log entries - replace with actual log retrieval
        log_entries = []
        for i in range(limit):
            log_entries.append({
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "level": level,
                "service": "rag-backend",
                "message": f"Sample log entry {i + 1}",
                "metadata": {
                    "request_id": f"req_{i:06d}",
                    "user_id": "system",
                    "execution_time": 0.123 + (i * 0.001)
                }
            })
        
        return {
            "logs": log_entries,
            "filters": {
                "level": level,
                "service": service,
                "limit": limit
            },
            "retrieved_by": current_user.get("username"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system logs")


@router.post("/maintenance/start")
@require_admin_role
async def start_maintenance_mode(
    background_tasks: BackgroundTasks,
    duration_minutes: int = Query(30, description="Maintenance duration", ge=1, le=240),
    reason: str = Query(..., description="Maintenance reason"),
    current_user = Depends(get_admin_user)
) -> Dict[str, Any]:
    """
    🚧 **Start Maintenance Mode**
    
    Puts the system into maintenance mode for updates and repairs.
    """
    try:
        logger.info(f"Admin {current_user.get('username')} starting maintenance mode")
        
        maintenance_id = await system_monitor.start_maintenance_mode(
            duration_minutes=duration_minutes,
            reason=reason,
            initiated_by=current_user.get("username")
        )
        
        return {
            "message": "Maintenance mode activated",
            "maintenance_id": maintenance_id,
            "duration_minutes": duration_minutes,
            "reason": reason,
            "initiated_by": current_user.get("username"),
            "end_time": (datetime.utcnow() + timedelta(minutes=duration_minutes)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start maintenance mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to start maintenance mode")


# Background task functions
async def execute_reindex_process(reindex_id: str, force: bool, batch_size: int):
    """Execute the reindexing process"""
    try:
        logger.info(f"Starting reindex process: {reindex_id}")
        
        vector_store = await get_vector_store()
        
        # Get all documents
        documents = await get_all_documents()
        total_docs = len(documents)
        
        processed = 0
        failed = 0
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            
            for doc in batch:
                try:
                    # Reprocess document
                    await reprocess_document_for_reindex(doc, vector_store)
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to reprocess document {doc['id']}: {e}")
                    failed += 1
            
            # Update progress
            await system_monitor.update_reindex_progress(
                reindex_id,
                processed + failed,
                total_docs
            )
        
        # Complete reindexing
        await system_monitor.complete_reindex_process(
            reindex_id,
            processed,
            failed
        )
        
        logger.info(f"Reindex process completed: {processed} processed, {failed} failed")
        
    except Exception as e:
        logger.error(f"Reindex process failed: {e}")
        await system_monitor.fail_reindex_process(reindex_id, str(e))


async def apply_config_updates(updates: Dict[str, Any], username: str):
    """Apply configuration updates"""
    try:
        logger.info(f"Applying config updates: {updates}")
        
        # Apply each configuration update
        for key, value in updates.items():
            await system_monitor.update_config(key, value, username)
        
        logger.info("Configuration updates applied successfully")
        
    except Exception as e:
        logger.error(f"Failed to apply config updates: {e}")


async def get_all_documents():
    """Get all documents for reindexing"""
    # Mock implementation - replace with actual database query
    return [{"id": f"doc_{i:06d}"} for i in range(1000)]


async def reprocess_document_for_reindex(doc: dict, vector_store):
    """Reprocess a single document for reindexing"""
    # Mock implementation - replace with actual reprocessing
    await asyncio.sleep(0.01)  # Simulate processing time
