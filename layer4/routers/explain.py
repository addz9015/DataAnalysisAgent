"""Explanation endpoints"""
from fastapi import APIRouter, HTTPException, Path

from ..models.responses import ExplainResponse
from ..dependencies import get_agent

router = APIRouter()

@router.get("/{claim_id}", response_model=ExplainResponse)
async def explain_claim(claim_id: str = Path(..., description="Claim ID")):
    """Get detailed explanation for a claim"""
    try:
        agent = get_agent()
        
        # Find in history
        history = [h for h in agent.decision_history if h['claim_id'] == claim_id]
        if not history:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        # Get full explanation from agent memory if available
        # For now, return basic info
        return ExplainResponse(
            claim_id=claim_id,
            summary=f"Claim {claim_id} was processed by the agent.",
            detailed="Detailed explanation would be retrieved from storage.",
            technical="{}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_claims(claim_ids: list):
    """Compare multiple claims"""
    return {"compared": claim_ids, "analysis": "Comparison results"}