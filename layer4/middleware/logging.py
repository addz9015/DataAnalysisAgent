# layer4/middleware/logging.py
"""
Request/response logging middleware
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

logger = logging.getLogger("layer4.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all API requests and responses
    """
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Start timer
        start_time = time.time()
        
        # Get request details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        path = request.url.path
        
        # Skip logging for health checks (optional)
        if path in ["/health", "/health/ready", "/health/live"]:
            return await call_next(request)
        
        # Log request
        logger.info(f"-> {method} {path} | IP: {client_ip}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            status_code = response.status_code
            status_emoji = "OK" if status_code < 400 else "WARN" if status_code < 500 else "FAIL"
            
            logger.info(
                f"<- {status_emoji} {method} {path} | "
                f"Status: {status_code} | "
                f"Duration: {duration:.3f}s"
            )
            
            # Add custom header with duration
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"FAIL {method} {path} | "
                f"Error: {str(e)} | "
                f"Duration: {duration:.3f}s"
            )
            raise


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Detailed error logging
    """
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await call_next(request)
            
        except Exception as e:
            # Log detailed error
            logger.exception(
                f"Unhandled exception in {request.method} {request.url.path}: {str(e)}"
            )
            raise


def setup_logging(log_level: str = "INFO", log_file: str = "logs/api.log"):
    """
    Setup logging configuration
    """
    import os
    from logging.handlers import RotatingFileHandler
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger("layer4")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # UVicorn access logger
    logging.getLogger("uvicorn.access").handlers = [console_handler]
    
    return root_logger