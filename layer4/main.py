# layer4/main.py
"""FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import predict, batch, explain, query, feedback, health
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware, ErrorLoggingMiddleware, setup_logging
from .config import settings

# Setup logging
logger = setup_logging(log_level="INFO")

app = FastAPI(
    title="StochClaim API",
    description="Autonomous Insurance Fraud Detection Agent API",
    version="1.0.0"
)

# Middleware (order matters - last added = first executed)
app.add_middleware(ErrorLoggingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.RATE_LIMIT,
    window=60
)
app.add_middleware(
    AuthMiddleware,
    api_key=settings.API_KEY
)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(batch.router, prefix="/batch", tags=["Batch"])
app.include_router(explain.router, prefix="/explain", tags=["Explanations"])
app.include_router(query.router, prefix="/query", tags=["Queries"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

@app.on_event("startup")
async def startup():
    logger.info("StochClaim API starting...")
    logger.info(f"   API Key: {'*' * 8}{settings.API_KEY[-4:]}")
    logger.info(f"   Rate Limit: {settings.RATE_LIMIT}/min")

@app.on_event("shutdown")
async def shutdown():
    logger.info("StochClaim API shutting down...")