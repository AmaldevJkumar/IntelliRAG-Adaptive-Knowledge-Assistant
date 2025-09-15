"""
Advanced logging configuration for RAG system
Structured logging with multiple outputs and formats
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from app.config import get_settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "exc_info", "exc_text", "stack_info",
                          "lineno", "funcName", "created", "msecs", "relativeCreated",
                          "thread", "threadName", "processName", "process", "getMessage"]:
                log_data["extra"] = log_data.get("extra", {})
                log_data["extra"][key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create formatted message
        formatted = f"{color}[{timestamp}] {record.levelname} {record.name}: {record.getMessage()}{self.RESET}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class RequestContextFilter(logging.Filter):
    """Add request context to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add request ID if available (would be set by middleware)
        if not hasattr(record, 'request_id'):
            record.request_id = getattr(self, '_request_id', 'no-request')
        
        # Add user ID if available
        if not hasattr(record, 'user_id'):
            record.user_id = getattr(self, '_user_id', 'anonymous')
        
        return True


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    json_logs: bool = None,
    max_log_size: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 5
):
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        json_logs: Use JSON formatting for structured logs
        max_log_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    try:
        settings = get_settings()
        
        # Set defaults from settings
        log_level = log_level or settings.LOG_LEVEL
        json_logs = json_logs if json_logs is not None else settings.is_production
        log_file = log_file or "logs/rag-assistant.log"
        
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if json_logs:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter()
        
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Add request context filter
        request_filter = RequestContextFilter()
        console_handler.addFilter(request_filter)
        
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Always use JSON format for file logs
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log all levels to file
        file_handler.addFilter(request_filter)
        
        root_logger.addHandler(file_handler)
        
        # Error file handler (errors only)
        error_file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path.parent / "errors.log"),
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(file_formatter)
        error_file_handler.addFilter(request_filter)
        
        root_logger.addHandler(error_file_handler)
        
        # Configure specific loggers
        configure_library_loggers()
        
        # Log startup message
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured: level={log_level}, json={json_logs}, file={log_file}")
        
    except Exception as e:
        print(f"Failed to setup logging: {e}", file=sys.stderr)
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def configure_library_loggers():
    """Configure logging for third-party libraries"""
    # Reduce noise from common libraries
    library_levels = {
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING,
        'asyncio': logging.WARNING,
        'multipart': logging.WARNING,
        'boto3': logging.WARNING,
        'botocore': logging.WARNING,
        'openai': logging.WARNING,
        'anthropic': logging.WARNING,
        'langchain': logging.INFO,
        'sentence_transformers': logging.WARNING,
        'transformers': logging.WARNING,
        'torch': logging.WARNING,
        'faiss': logging.WARNING,
        'pinecone': logging.INFO,
        'redis': logging.WARNING
    }
    
    for logger_name, level in library_levels.items():
        logging.getLogger(logger_name).setLevel(level)


class LoggerMixin:
    """Mixin class to add logging capabilities"""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


class StructuredLogger:
    """Structured logger with context support"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log message with context"""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        kwargs.setdefault('exc_info', True)
        self.error(message, *args, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


# Request logging middleware would use these functions
def set_request_context(request_id: str, user_id: str = None):
    """Set request context for logging"""
    # This would be called by middleware to set context
    pass


def clear_request_context():
    """Clear request context"""
    # This would be called at the end of request
    pass


# Performance logging utilities
class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_query_performance(
        self,
        query_id: str,
        query: str,
        response_time: float,
        retrieval_time: float,
        generation_time: float,
        confidence_score: float,
        num_documents: int
    ):
        """Log query performance metrics"""
        self.logger.info(
            "Query performance metrics",
            extra={
                "query_id": query_id,
                "query_length": len(query),
                "response_time_ms": response_time * 1000,
                "retrieval_time_ms": retrieval_time * 1000,
                "generation_time_ms": generation_time * 1000,
                "confidence_score": confidence_score,
                "num_documents_retrieved": num_documents,
                "metric_type": "query_performance"
            }
        )
    
    def log_document_processing(
        self,
        document_id: str,
        filename: str,
        processing_time: float,
        chunks_created: int,
        file_size: int,
        success: bool
    ):
        """Log document processing metrics"""
        self.logger.info(
            "Document processing metrics",
            extra={
                "document_id": document_id,
                "filename": filename,
                "processing_time_seconds": processing_time,
                "chunks_created": chunks_created,
                "file_size_bytes": file_size,
                "success": success,
                "metric_type": "document_processing"
            }
        )


# Audit logging
class AuditLogger:
    """Logger for security and audit events"""
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        # Set to always log audit events
        self.logger.setLevel(logging.INFO)
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str = None):
        """Log authentication attempt"""
        self.logger.info(
            "Authentication attempt",
            extra={
                "user_id": user_id,
                "success": success,
                "ip_address": ip_address,
                "event_type": "authentication",
                "severity": "medium" if success else "high"
            }
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, allowed: bool):
        """Log authorization check"""
        self.logger.info(
            "Authorization check",
            extra={
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "allowed": allowed,
                "event_type": "authorization",
                "severity": "low" if allowed else "medium"
            }
        )
    
    def log_data_access(self, user_id: str, document_id: str, query: str):
        """Log data access"""
        self.logger.info(
            "Data access",
            extra={
                "user_id": user_id,
                "document_id": document_id,
                "query": query[:100],  # Truncate for privacy
                "event_type": "data_access",
                "severity": "low"
            }
        )
    
    def log_configuration_change(self, user_id: str, setting: str, old_value: Any, new_value: Any):
        """Log configuration change"""
        self.logger.warning(
            "Configuration change",
            extra={
                "user_id": user_id,
                "setting": setting,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "event_type": "configuration_change",
                "severity": "high"
            }
        )
