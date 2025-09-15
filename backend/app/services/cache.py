"""
Caching service for RAG system
Redis-based caching with fallback to in-memory
"""

import logging
import asyncio
import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.config import get_settings
from app.utils.metrics import increment_counter

logger = logging.getLogger(__name__)


class CacheService:
    """Advanced caching service with multiple backends"""
    
    def __init__(self, redis_url: str = None):
        self.settings = get_settings()
        self.redis_url = redis_url or self.settings.REDIS_URL
        self.redis_client = None
        self.fallback_cache = {}  # In-memory fallback
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def initialize(self):
        """Initialize cache connections"""
        try:
            if REDIS_AVAILABLE and self.redis_url:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            else:
                logger.warning("Redis not available, using in-memory fallback cache")
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            logger.info("Using in-memory fallback cache")
            self.redis_client = None
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value is not None:
                    self.cache_stats["hits"] += 1
                    increment_counter("cache_hits_total", {"backend": "redis"})
                    return json.loads(value)
            else:
                # Fallback to in-memory cache
                if key in self.fallback_cache:
                    entry = self.fallback_cache[key]
                    if entry["expires_at"] > datetime.utcnow():
                        self.cache_stats["hits"] += 1
                        increment_counter("cache_hits_total", {"backend": "memory"})
                        return entry["value"]
                    else:
                        # Expired entry
                        del self.fallback_cache[key]
            
            # Cache miss
            self.cache_stats["misses"] += 1
            increment_counter("cache_misses_total")
            return default
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None,
        nx: bool = False
    ) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.settings.CACHE_TTL
            
            if self.redis_client:
                serialized_value = json.dumps(value, default=str)
                result = await self.redis_client.set(
                    key, 
                    serialized_value, 
                    ex=ttl,
                    nx=nx
                )
                success = result is not False
            else:
                # Fallback to in-memory cache
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                self.fallback_cache[key] = {
                    "value": value,
                    "expires_at": expires_at
                }
                success = True
            
            if success:
                self.cache_stats["sets"] += 1
                increment_counter("cache_sets_total")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                result = await self.redis_client.delete(key)
                success = result > 0
            else:
                success = key in self.fallback_cache
                if success:
                    del self.fallback_cache[key]
            
            if success:
                self.cache_stats["deletes"] += 1
                increment_counter("cache_deletes_total")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        try:
            deleted_count = 0
            
            if self.redis_client:
                # Use Redis SCAN for pattern matching
                async for key in self.redis_client.scan_iter(match=pattern):
                    if await self.redis_client.delete(key):
                        deleted_count += 1
            else:
                # Fallback pattern matching
                import fnmatch
                keys_to_delete = [
                    key for key in self.fallback_cache.keys()
                    if fnmatch.fnmatch(key, pattern)
                ]
                for key in keys_to_delete:
                    del self.fallback_cache[key]
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} keys matching pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache pattern delete failed for pattern {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.redis_client:
                return await self.redis_client.exists(key) > 0
            else:
                if key in self.fallback_cache:
                    entry = self.fallback_cache[key]
                    if entry["expires_at"] > datetime.utcnow():
                        return True
                    else:
                        del self.fallback_cache[key]
                return False
                
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys from cache"""
        try:
            result = {}
            
            if self.redis_client:
                values = await self.redis_client.mget(keys)
                for key, value in zip(keys, values):
                    if value is not None:
                        result[key] = json.loads(value)
                        self.cache_stats["hits"] += 1
                    else:
                        self.cache_stats["misses"] += 1
            else:
                for key in keys:
                    value = await self.get(key)
                    if value is not None:
                        result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_many failed: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple key-value pairs"""
        try:
            ttl = ttl or self.settings.CACHE_TTL
            
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                for key, value in mapping.items():
                    serialized_value = json.dumps(value, default=str)
                    pipe.set(key, serialized_value, ex=ttl)
                
                results = await pipe.execute()
                success = all(results)
            else:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                for key, value in mapping.items():
                    self.fallback_cache[key] = {
                        "value": value,
                        "expires_at": expires_at
                    }
                success = True
            
            if success:
                self.cache_stats["sets"] += len(mapping)
                increment_counter("cache_sets_total", {"count": len(mapping)})
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set_many failed: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        try:
            if self.redis_client:
                return await self.redis_client.incr(key, amount)
            else:
                current = await self.get(key, 0)
                new_value = int(current) + amount
                await self.set(key, new_value)
                return new_value
                
        except Exception as e:
            logger.error(f"Cache increment failed for key {key}: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        try:
            if self.redis_client:
                return await self.redis_client.expire(key, ttl)
            else:
                if key in self.fallback_cache:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                    self.fallback_cache[key]["expires_at"] = expires_at
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache expire failed for key {key}: {e}")
            return False
    
    async def flushall(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                await self.redis_client.flushall()
            else:
                self.fallback_cache.clear()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache flush failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = dict(self.cache_stats)
            
            if self.redis_client:
                info = await self.redis_client.info()
                stats.update({
                    "backend": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                })
            else:
                stats.update({
                    "backend": "memory",
                    "total_keys": len(self.fallback_cache),
                    "expired_keys": sum(
                        1 for entry in self.fallback_cache.values()
                        if entry["expires_at"] <= datetime.utcnow()
                    )
                })
            
            # Calculate hit rate
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return self.cache_stats
    
    async def cleanup_expired(self):
        """Clean up expired entries (for in-memory cache)"""
        if not self.redis_client:  # Only needed for in-memory cache
            try:
                current_time = datetime.utcnow()
                expired_keys = [
                    key for key, entry in self.fallback_cache.items()
                    if entry["expires_at"] <= current_time
                ]
                
                for key in expired_keys:
                    del self.fallback_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and parameters"""
        # Create a deterministic key from arguments
        key_parts = [prefix]
        
        # Add positional arguments
        key_parts.extend(str(arg) for arg in args)
        
        # Add keyword arguments (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        
        # Create hash for very long keys
        key_string = ":".join(key_parts)
        if len(key_string) > 200:  # Redis key length limit consideration
            hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
            key_string = f"{prefix}:hash:{hash_suffix}"
        
        return key_string


# Global cache instance
_cache_service = None


async def get_cache() -> CacheService:
    """Get the global cache service instance"""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.initialize()
    
    return _cache_service


async def initialize_cache():
    """Initialize the cache service on startup"""
    await get_cache()
    logger.info("Cache service initialized")


# Cache decorators for easy use
def cached(ttl: int = None, key_prefix: str = "func"):
    """Decorator to cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = await get_cache()
            
            # Generate cache key
            cache_key = cache.generate_key(
                f"{key_prefix}:{func.__name__}",
                *args,
                **kwargs
            )
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator
