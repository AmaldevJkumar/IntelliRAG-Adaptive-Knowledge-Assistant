"""
Security utilities for RAG system
Authentication, authorization, and security helpers
"""

import logging
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handling
security = HTTPBearer()

# Rate limiting storage (in production, use Redis)
_rate_limit_storage = {}


class SecurityUtils:
    """Security utility functions"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
        """Generate JWT token"""
        settings = get_settings()
        
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            settings = get_settings()
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not text:
            return ""
        
        # Basic sanitization - remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r\n', '\r', '\n']
        sanitized = text
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        return sanitized[:1000]
    
    @staticmethod
    def validate_query_safety(query: str) -> bool:
        """Validate that query is safe to process"""
        if not query or len(query) > 2000:
            return False
        
        # Check for potential injection patterns
        dangerous_patterns = [
            'javascript:', 'data:', 'vbscript:', '<script', '</script>',
            'on\w+\s*=', 'eval\s*\(', 'function\s*\(', 'window\.',
            'document\.', 'alert\s*\(', 'prompt\s*\(', 'confirm\s*\('
        ]
        
        import re
        query_lower = query.lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Potentially dangerous query pattern detected: {pattern}")
                return False
        
        return True


class RateLimiter:
    """Simple rate limiting implementation"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.utcnow()
        
        if identifier not in _rate_limit_storage:
            _rate_limit_storage[identifier] = []
        
        # Clean old requests
        cutoff = now - timedelta(seconds=self.window_seconds)
        _rate_limit_storage[identifier] = [
            req_time for req_time in _rate_limit_storage[identifier]
            if req_time > cutoff
        ]
        
        # Check if under limit
        if len(_rate_limit_storage[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        _rate_limit_storage[identifier].append(now)
        return True
    
    def get_reset_time(self, identifier: str) -> Optional[datetime]:
        """Get when rate limit resets for identifier"""
        if identifier not in _rate_limit_storage:
            return None
        
        if not _rate_limit_storage[identifier]:
            return None
        
        oldest_request = min(_rate_limit_storage[identifier])
        return oldest_request + timedelta(seconds=self.window_seconds)


# Global rate limiter instance
default_rate_limiter = RateLimiter()


class User:
    """User model for authentication"""
    
    def __init__(self, user_id: str, username: str, email: str, roles: List[str] = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles or ["user"]
        self.is_active = True
        self.is_admin = "admin" in self.roles


class Permission:
    """Permission constants"""
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    QUERY_SYSTEM = "query_system"
    ADMIN_ACCESS = "admin_access"
    SYSTEM_CONFIG = "system_config"


class RolePermissions:
    """Role-based permission mapping"""
    
    PERMISSIONS = {
        "user": [
            Permission.READ_DOCUMENTS,
            Permission.QUERY_SYSTEM
        ],
        "editor": [
            Permission.READ_DOCUMENTS,
            Permission.WRITE_DOCUMENTS,
            Permission.QUERY_SYSTEM
        ],
        "admin": [
            Permission.READ_DOCUMENTS,
            Permission.WRITE_DOCUMENTS,
            Permission.DELETE_DOCUMENTS,
            Permission.QUERY_SYSTEM,
            Permission.ADMIN_ACCESS,
            Permission.SYSTEM_CONFIG
        ]
    }
    
    @classmethod
    def has_permission(cls, roles: List[str], permission: str) -> bool:
        """Check if roles have specific permission"""
        for role in roles:
            if role in cls.PERMISSIONS and permission in cls.PERMISSIONS[role]:
                return True
        return False


# Mock user database (in production, use actual database)
MOCK_USERS = {
    "admin": {
        "user_id": "admin_001",
        "username": "admin",
        "email": "admin@ragassistant.com",
        "hashed_password": SecurityUtils.hash_password("admin123"),
        "roles": ["admin"],
        "api_key": SecurityUtils.hash_api_key("sk-admin-key-12345")
    },
    "user": {
        "user_id": "user_001", 
        "username": "user",
        "email": "user@ragassistant.com",
        "hashed_password": SecurityUtils.hash_password("user123"),
        "roles": ["user"],
        "api_key": SecurityUtils.hash_api_key("sk-user-key-67890")
    }
}


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    try:
        # Verify token
        payload = SecurityUtils.verify_token(credentials.credentials)
        username = payload.get("sub")
        
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        
        # Get user from database (mock)
        if username not in MOCK_USERS:
            raise HTTPException(status_code=401, detail="User not found")
        
        user_data = MOCK_USERS[username]
        
        return User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data["roles"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current user with admin privileges"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Not enough permissions"
        )
    return current_user


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, current_user: User = Depends(get_current_active_user), **kwargs):
            if not RolePermissions.has_permission(current_user.roles, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission required: {permission}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_admin_role(func):
    """Decorator to require admin role"""
    async def wrapper(*args, current_user: User = Depends(get_admin_user), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper


class IPWhitelist:
    """IP address whitelisting"""
    
    def __init__(self, allowed_ips: List[str] = None):
        self.allowed_ips = set(allowed_ips or [])
        # Add localhost by default
        self.allowed_ips.update(["127.0.0.1", "::1", "localhost"])
    
    def is_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed"""
        if not self.allowed_ips:  # If no whitelist, allow all
            return True
        
        return ip_address in self.allowed_ips
    
    def add_ip(self, ip_address: str):
        """Add IP to whitelist"""
        self.allowed_ips.add(ip_address)
    
    def remove_ip(self, ip_address: str):
        """Remove IP from whitelist"""
        self.allowed_ips.discard(ip_address)


# Rate limiting dependency
def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate limiting"""
    client_ip = request.client.host
    
    if not default_rate_limiter.is_allowed(client_ip):
        reset_time = default_rate_limiter.get_reset_time(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Rate limit exceeded",
                "reset_time": reset_time.isoformat() if reset_time else None
            }
        )


class SecurityHeaders:
    """Security headers middleware"""
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


def authenticate_api_key(api_key: str) -> Optional[User]:
    """Authenticate using API key"""
    if not api_key:
        return None
    
    hashed_key = SecurityUtils.hash_api_key(api_key)
    
    for username, user_data in MOCK_USERS.items():
        if user_data.get("api_key") == hashed_key:
            return User(
                user_id=user_data["user_id"],
                username=username,
                email=user_data["email"],
                roles=user_data["roles"]
            )
    
    return None


def authenticate_user(username: str, password: str) -> Optional[str]:
    """Authenticate user credentials and return token"""
    if username not in MOCK_USERS:
        return None
    
    user_data = MOCK_USERS[username]
    
    if not SecurityUtils.verify_password(password, user_data["hashed_password"]):
        return None
    
    # Generate token
    token_data = {"sub": username, "roles": user_data["roles"]}
    access_token = SecurityUtils.generate_token(token_data)
    
    return access_token


# Input validation
class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username"""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        
        import re
        pattern = r'^[a-zA-Z0-9_-]+$'
        return re.match(pattern, username) is not None
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {
            "valid": True,
            "score": 0,
            "issues": []
        }
        
        if len(password) < 8:
            result["issues"].append("Password must be at least 8 characters long")
            result["valid"] = False
        else:
            result["score"] += 1
        
        if not any(c.isupper() for c in password):
            result["issues"].append("Password must contain at least one uppercase letter")
            result["valid"] = False
        else:
            result["score"] += 1
        
        if not any(c.islower() for c in password):
            result["issues"].append("Password must contain at least one lowercase letter")
            result["valid"] = False
        else:
            result["score"] += 1
        
        if not any(c.isdigit() for c in password):
            result["issues"].append("Password must contain at least one digit")
            result["valid"] = False
        else:
            result["score"] += 1
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            result["issues"].append("Password must contain at least one special character")
            result["valid"] = False
        else:
            result["score"] += 1
        
        return result


# Security audit logging
def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security event for audit"""
    from app.utils.logging import AuditLogger
    
    audit_logger = AuditLogger()
    
    logger.warning(
        f"Security event: {event_type}",
        extra={
            "event_type": event_type,
            "details": details,
            "severity": "high"
        }
    )
