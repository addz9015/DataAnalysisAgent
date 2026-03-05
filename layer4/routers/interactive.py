"""Interactive tools via API"""
from fastapi import APIRouter, Depends
from ..interactive.quick_check import is_fraud, quick_check
from ..interactive.cli import interactive_cli
from ..config import settings

router = APIRouter()

@router.post("/quick-check")
async def api_quick_check(claim_data: dict):
    """Quick fraud check via API"""
    if not settings.INTERACTIVE_ENABLED:
        return {"error": "Interactive mode disabled"}
    return quick_check(claim_data)

@router.get("/cli")
async def run_cli():
    """Trigger interactive CLI (for local use)"""
    if not settings.INTERACTIVE_ENABLED:
        return {"error": "Interactive mode disabled"}
    interactive_cli()
    return {"status": "CLI completed"}