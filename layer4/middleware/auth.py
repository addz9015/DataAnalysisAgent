"""Authentication middleware"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger("layer4.auth")

class AuthMiddleware(BaseHTTPMiddleware):
    """API key authentication"""
    
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key
    
    async def dispatch(self, request: Request, call_next):
        # Skip health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != self.api_key:
            logger.warning(f"Invalid API key from {request.client.host}")
            raise HTTPException(status_code=403, detail="Invalid or missing API key")
        
        return await call_next(request)