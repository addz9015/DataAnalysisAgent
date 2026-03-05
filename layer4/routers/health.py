"""Health check endpoints"""
from fastapi import APIRouter
from datetime import datetime
from ..models.responses import HealthResponse

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        timestamp=datetime.now().isoformat()
    )