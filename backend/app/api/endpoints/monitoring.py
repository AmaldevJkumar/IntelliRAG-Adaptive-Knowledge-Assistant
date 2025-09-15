"""
Monitoring and analytics endpoints
Performance tracking and quality assessment
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from app.models.schemas import (
    EvaluationRequest, EvaluationResult, BatchEvaluationRequest, 
    BatchEvaluationResponse, SystemMetrics
)
from app.services.monitoring import QualityMonitor, PerformanceTracker
from app.services.evaluation import RAGEvaluator
from app.utils.metrics import get_prometheus_metrics, export_metrics
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

quality_monitor = QualityMonitor()
performance_tracker = PerformanceTracker()
rag_evaluator = RAGEvaluator()
settings = get_settings()


class MetricsResponse(BaseModel):
    """Metrics response model"""
    timestamp: datetime
    time_range: str
    metrics: Dict[str, Any]


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics() -> SystemMetrics:
    """
    📊 **Get System Performance Metrics**
    
    Retrieves comprehensive system performance metrics including:
    - Query processing statistics
    - Response time distributions
    - Confidence score analytics
    - Cache performance metrics
    - Resource utilization data
    """
    try:
        logger.info("Retrieving system metrics")
        
        # Get metrics from performance tracker
        metrics_data = await performance_tracker.get_current_metrics()
        
        return SystemMetrics(
            uptime_hours=metrics_data.get("uptime_hours", 0.0),
            total_queries=metrics_data.get("total_queries", 0),
            avg_response_time_ms=metrics_data.get("avg_response_time_ms", 0.0),
            avg_confidence_score=metrics_data.get("avg_confidence_score", 0.0),
            cache_hit_rate=metrics_data.get("cache_hit_rate", 0.0),
            active_documents=metrics_data.get("active_documents", 0),
            vector_db_status=metrics_data.get("vector_db_status", "unknown"),
            llm_service_status=metrics_data.get("llm_service_status", "unknown"),
            memory_usage_mb=metrics_data.get("memory_usage_mb", 0.0),
            cpu_usage_percent=metrics_data.get("cpu_usage_percent", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/metrics/detailed", response_model=MetricsResponse)
async def get_detailed_metrics(
    time_range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    metric_type: str = Query("all", description="Metric type filter")
) -> MetricsResponse:
    """
    📈 **Get Detailed Performance Metrics**
    
    Provides in-depth performance analytics with time-series data,
    histograms, and trend analysis.
    """
    try:
        logger.info(f"Retrieving detailed metrics for {time_range}")
        
        # Parse time range
        time_delta_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7)
        }
        
        if time_range not in time_delta_map:
            raise HTTPException(status_code=400, detail="Invalid time range")
        
        end_time = datetime.utcnow()
        start_time = end_time - time_delta_map[time_range]
        
        # Get detailed metrics
        detailed_metrics = await performance_tracker.get_detailed_metrics(
            start_time=start_time,
            end_time=end_time,
            metric_type=metric_type
        )
        
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            time_range=time_range,
            metrics=detailed_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve detailed metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve detailed metrics")


@router.get("/quality/dashboard")
async def get_quality_dashboard() -> Dict[str, Any]:
    """
    🎯 **Quality Assessment Dashboard**
    
    Provides comprehensive quality metrics and insights:
    - Answer quality scores and trends
    - Retrieval accuracy metrics
    - User satisfaction ratings
    - Error analysis and patterns
    """
    try:
        logger.info("Retrieving quality dashboard data")
        
        # Get quality metrics from the last 24 hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        quality_data = await quality_monitor.get_quality_dashboard(
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_range": "24h",
            
            "overview": {
                "total_queries": quality_data.get("total_queries", 0),
                "avg_quality_score": quality_data.get("avg_quality_score", 0.0),
                "user_satisfaction": quality_data.get("user_satisfaction", 0.0),
                "error_rate": quality_data.get("error_rate", 0.0),
                "improvement_trend": quality_data.get("improvement_trend", "stable")
            },
            
            "quality_metrics": {
                "answer_relevancy": {
                    "score": quality_data.get("answer_relevancy", 0.0),
                    "trend": quality_data.get("answer_relevancy_trend", "stable"),
                    "samples": quality_data.get("answer_relevancy_samples", 0)
                },
                "faithfulness": {
                    "score": quality_data.get("faithfulness", 0.0),
                    "trend": quality_data.get("faithfulness_trend", "stable"),
                    "samples": quality_data.get("faithfulness_samples", 0)
                },
                "context_precision": {
                    "score": quality_data.get("context_precision", 0.0),
                    "trend": quality_data.get("context_precision_trend", "stable"),
                    "samples": quality_data.get("context_precision_samples", 0)
                },
                "context_recall": {
                    "score": quality_data.get("context_recall", 0.0),
                    "trend": quality_data.get("context_recall_trend", "stable"),
                    "samples": quality_data.get("context_recall_samples", 0)
                }
            },
            
            "user_feedback": {
                "total_feedback": quality_data.get("total_feedback", 0),
                "positive_rate": quality_data.get("positive_feedback_rate", 0.0),
                "avg_rating": quality_data.get("avg_rating", 0.0),
                "common_issues": quality_data.get("common_issues", [])
            },
            
            "retrieval_performance": {
                "avg_retrieval_time": quality_data.get("avg_retrieval_time", 0.0),
                "avg_documents_retrieved": quality_data.get("avg_documents_retrieved", 0.0),
                "cache_hit_rate": quality_data.get("cache_hit_rate", 0.0),
                "reranking_effectiveness": quality_data.get("reranking_effectiveness", 0.0)
            },
            
            "alerts": quality_data.get("active_alerts", []),
            "recommendations": quality_data.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve quality dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality dashboard")


@router.post("/evaluate/single", response_model=EvaluationResult)
async def evaluate_single_query(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
) -> EvaluationResult:
    """
    🔍 **Evaluate Single Query**
    
    Performs comprehensive evaluation of a single query-answer pair
    using multiple quality metrics and frameworks.
    """
    try:
        logger.info(f"Evaluating single query: {request.query[:50]}...")
        
        # Perform evaluation
        evaluation_result = await rag_evaluator.evaluate_single_query(request)
        
        # Log evaluation for tracking
        background_tasks.add_task(
            log_evaluation_result,
            evaluation_result
        )
        
        return evaluation_result
        
    except Exception as e:
        logger.error(f"Single query evaluation failed: {e}")
        raise HTTPException(status_code=500, detail="Query evaluation failed")


@router.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch_queries(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks
) -> BatchEvaluationResponse:
    """
    📊 **Batch Query Evaluation**
    
    Evaluates multiple queries in batch for comprehensive
    system assessment and benchmarking.
    """
    try:
        logger.info(f"Starting batch evaluation: {request.run_name}")
        
        # Start batch evaluation
        evaluation_response = await rag_evaluator.start_batch_evaluation(request)
        
        # Process evaluation asynchronously
        background_tasks.add_task(
            process_batch_evaluation,
            evaluation_response.run_id,
            request
        )
        
        return evaluation_response
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail="Batch evaluation failed")


@router.get("/evaluate/batch/{run_id}")
async def get_evaluation_results(run_id: UUID) -> Dict[str, Any]:
    """
    📋 **Get Evaluation Results**
    
    Retrieves results from a batch evaluation run.
    """
    try:
        logger.info(f"Retrieving evaluation results for run: {run_id}")
        
        results = await rag_evaluator.get_evaluation_results(run_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation results")


@router.get("/performance/trends")
async def get_performance_trends(
    metric: str = Query("response_time", description="Metric to analyze"),
    time_range: str = Query("7d", description="Time range for trend analysis")
) -> Dict[str, Any]:
    """
    📈 **Performance Trend Analysis**
    
    Analyzes performance trends over time with statistical insights.
    """
    try:
        logger.info(f"Analyzing performance trends for {metric}")
        
        # Get trend data
        trend_data = await performance_tracker.get_performance_trends(
            metric=metric,
            time_range=time_range
        )
        
        return {
            "metric": metric,
            "time_range": time_range,
            "trend_analysis": trend_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Performance trend analysis failed")


@router.get("/alerts")
async def get_active_alerts() -> Dict[str, Any]:
    """
    🚨 **Get Active Alerts**
    
    Retrieves current system alerts and notifications.
    """
    try:
        logger.info("Retrieving active alerts")
        
        alerts = await quality_monitor.get_active_alerts()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": alerts,
            "alert_counts": {
                "critical": len([a for a in alerts if a.get("severity") == "critical"]),
                "warning": len([a for a in alerts if a.get("severity") == "warning"]),
                "info": len([a for a in alerts if a.get("severity") == "info"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> Dict[str, Any]:
    """
    ✅ **Acknowledge Alert**
    
    Acknowledges an active alert to prevent notification spam.
    """
    try:
        logger.info(f"Acknowledging alert: {alert_id}")
        
        result = await quality_monitor.acknowledge_alert(alert_id)
        
        return {
            "message": "Alert acknowledged successfully",
            "alert_id": alert_id,
            "acknowledged_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.get("/export/metrics")
async def export_metrics_data(
    format_type: str = Query("json", description="Export format: json, csv, prometheus"),
    time_range: str = Query("24h", description="Time range for export")
) -> Dict[str, Any]:
    """
    📤 **Export Metrics Data**
    
    Exports system metrics in various formats for external analysis.
    """
    try:
        logger.info(f"Exporting metrics data: format={format_type}, range={time_range}")
        
        export_data = await export_metrics(
            format_type=format_type,
            time_range=time_range
        )
        
        return {
            "format": format_type,
            "time_range": time_range,
            "export_timestamp": datetime.utcnow().isoformat(),
            "data": export_data
        }
        
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics export failed")


# Background task functions
async def log_evaluation_result(evaluation_result: EvaluationResult):
    """Log evaluation result for tracking"""
    try:
        await quality_monitor.log_evaluation_result(evaluation_result)
        logger.info(f"Logged evaluation result for query: {evaluation_result.query[:50]}")
        
    except Exception as e:
        logger.error(f"Failed to log evaluation result: {e}")


async def process_batch_evaluation(run_id: UUID, request: BatchEvaluationRequest):
    """Process batch evaluation asynchronously"""
    try:
        logger.info(f"Processing batch evaluation: {run_id}")
        
        # Process each query in the batch
        results = []
        for query_request in request.queries:
            try:
                result = await rag_evaluator.evaluate_single_query(query_request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate query: {query_request.query[:50]}: {e}")
        
        # Update batch evaluation with results
        await rag_evaluator.complete_batch_evaluation(run_id, results)
        
        logger.info(f"Batch evaluation completed: {run_id}")
        
    except Exception as e:
        logger.error(f"Batch evaluation processing failed: {e}")
        await rag_evaluator.fail_batch_evaluation(run_id, str(e))
