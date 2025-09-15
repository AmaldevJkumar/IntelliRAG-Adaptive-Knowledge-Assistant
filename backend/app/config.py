"""
Configuration management for RAG Knowledge Assistant
Centralized settings with environment variable support
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, validator, Field


class Settings(BaseSettings):
    """Application settings with comprehensive configuration options"""
    
    # Application Configuration
    APP_NAME: str = Field("RAG Knowledge Assistant", description="Application name")
    APP_VERSION: str = Field("1.0.0", description="Application version")
    DEBUG: bool = Field(False, description="Debug mode")
    HOST: str = Field("0.0.0.0", description="Server host")
    PORT: int = Field(8000, description="Server port")
    API_PREFIX: str = Field("/api/v1", description="API prefix")
    
    # Security Configuration
    SECRET_KEY: str = Field("change-this-in-production", description="Secret key for JWT")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(43200, description="Token expiration time")
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        "postgresql://rag_user:rag_password@localhost:5432/rag_db",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(20, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(0, description="Database max overflow connections")
    
    # Redis Cache Configuration
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    REDIS_POOL_SIZE: int = Field(10, description="Redis connection pool size")
    CACHE_TTL: int = Field(3600, description="Cache TTL in seconds")
    
    # Vector Database Configuration
    VECTOR_DB_TYPE: str = Field("faiss", description="Vector database type")
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = Field(None, description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = Field("us-west1-gcp-free", description="Pinecone environment")
    PINECONE_INDEX_NAME: str = Field("rag-knowledge-base", description="Pinecone index name")
    PINECONE_DIMENSION: int = Field(384, description="Pinecone vector dimension")
    PINECONE_METRIC: str = Field("cosine", description="Pinecone similarity metric")
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = Field("./data/faiss_index", description="FAISS index storage path")
    
    # LLM Configuration
    LLM_PROVIDER: str = Field("openai", description="LLM provider")
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    OPENAI_MODEL: str = Field("gpt-4", description="OpenAI model name")
    OPENAI_TEMPERATURE: float = Field(0.1, description="OpenAI temperature")
    OPENAI_MAX_TOKENS: int = Field(1000, description="OpenAI max tokens")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic API key")
    ANTHROPIC_MODEL: str = Field("claude-3-sonnet-20240229", description="Anthropic model")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", 
        description="Embedding model name"
    )
    EMBEDDING_BATCH_SIZE: int = Field(32, description="Embedding batch size")
    EMBEDDING_MAX_LENGTH: int = Field(512, description="Embedding max sequence length")
    
    # Document Processing Configuration
    CHUNK_SIZE: int = Field(1000, description="Document chunk size")
    CHUNK_OVERLAP: int = Field(200, description="Document chunk overlap")
    MAX_FILE_SIZE: int = Field(52428800, description="Max file size in bytes (50MB)")
    SUPPORTED_FILE_TYPES: str = Field("pdf,docx,txt,md,html", description="Supported file types")
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(5, description="Default number of documents to retrieve")
    RETRIEVAL_SCORE_THRESHOLD: float = Field(0.7, description="Minimum relevance score")
    HYBRID_SEARCH_ALPHA: float = Field(0.5, description="Hybrid search weighting")
    RERANK_TOP_K: int = Field(20, description="Number of documents for reranking")
    RERANK_MODEL: str = Field("ms-marco-MiniLM-L-6-v2", description="Reranking model")
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field("http://localhost:5000", description="MLflow tracking URI")
    MLFLOW_EXPERIMENT_NAME: str = Field("rag-knowledge-assistant", description="MLflow experiment")
    ENABLE_MLFLOW: bool = Field(True, description="Enable MLflow tracking")
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(True, description="Enable Prometheus metrics")
    ENABLE_FEEDBACK: bool = Field(True, description="Enable user feedback collection")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    QUALITY_THRESHOLD: float = Field(0.8, description="Quality threshold for alerts")
    AUTO_RETRAIN_THRESHOLD: float = Field(0.6, description="Auto-retrain threshold")
    DRIFT_DETECTION_WINDOW: int = Field(1000, description="Drift detection window size")
    
    # Performance Configuration
    MAX_WORKERS: int = Field(4, description="Maximum worker processes")
    REQUEST_TIMEOUT: int = Field(30, description="Request timeout in seconds")
    RATE_LIMIT_REQUESTS: int = Field(100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(60, description="Rate limit window in seconds")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(["*"], description="CORS allowed origins")
    ALLOWED_HOSTS: List[str] = Field(["*"], description="Allowed hosts")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator('DATABASE_URL', pre=True)
    def validate_database_url(cls, v):
        if v and not v.startswith(('postgresql://', 'sqlite://', 'mysql://')):
            raise ValueError('DATABASE_URL must be a valid database URL')
        return v
    
    @validator('VECTOR_DB_TYPE')
    def validate_vector_db_type(cls, v):
        allowed_types = ['pinecone', 'faiss', 'weaviate']
        if v not in allowed_types:
            raise ValueError(f'VECTOR_DB_TYPE must be one of {allowed_types}')
        return v
    
    @validator('LLM_PROVIDER')
    def validate_llm_provider(cls, v):
        allowed_providers = ['openai', 'anthropic', 'huggingface']
        if v not in allowed_providers:
            raise ValueError(f'LLM_PROVIDER must be one of {allowed_providers}')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'LOG_LEVEL must be one of {allowed_levels}')
        return v.upper()
    
    @property
    def supported_file_types_list(self) -> List[str]:
        """Get supported file types as a list"""
        return [ft.strip() for ft in self.SUPPORTED_FILE_TYPES.split(',')]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.DEBUG
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'url': self.DATABASE_URL,
            'pool_size': self.DATABASE_POOL_SIZE,
            'max_overflow': self.DATABASE_MAX_OVERFLOW
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            'url': self.REDIS_URL,
            'pool_size': self.REDIS_POOL_SIZE,
            'ttl': self.CACHE_TTL
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration"""
        if self.VECTOR_DB_TYPE == 'pinecone':
            return {
                'type': 'pinecone',
                'api_key': self.PINECONE_API_KEY,
                'environment': self.PINECONE_ENVIRONMENT,
                'index_name': self.PINECONE_INDEX_NAME,
                'dimension': self.PINECONE_DIMENSION,
                'metric': self.PINECONE_METRIC
            }
        elif self.VECTOR_DB_TYPE == 'faiss':
            return {
                'type': 'faiss',
                'index_path': self.FAISS_INDEX_PATH
            }
        else:
            return {'type': self.VECTOR_DB_TYPE}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        if self.LLM_PROVIDER == 'openai':
            return {
                'provider': 'openai',
                'api_key': self.OPENAI_API_KEY,
                'model': self.OPENAI_MODEL,
                'temperature': self.OPENAI_TEMPERATURE,
                'max_tokens': self.OPENAI_MAX_TOKENS
            }
        elif self.LLM_PROVIDER == 'anthropic':
            return {
                'provider': 'anthropic',
                'api_key': self.ANTHROPIC_API_KEY,
                'model': self.ANTHROPIC_MODEL
            }
        else:
            return {'provider': self.LLM_PROVIDER}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENABLE_MLFLOW: bool = True
    CORS_ORIGINS: List[str] = ["*"]


class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = []  # Should be configured properly
    
    @validator('SECRET_KEY')
    def validate_production_secret_key(cls, v):
        if v == "change-this-in-production":
            raise ValueError('SECRET_KEY must be changed in production')
        return v


class TestingSettings(Settings):
    """Testing environment settings"""
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"
    ENABLE_METRICS: bool = False
    ENABLE_MLFLOW: bool = False


def get_settings_for_env(env: str = None) -> Settings:
    """Get settings based on environment"""
    env = env or os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()
