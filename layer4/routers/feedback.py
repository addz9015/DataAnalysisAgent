"""Human feedback endpoints"""
from fastapi import APIRouter, HTTPException

from ..models.requests import FeedbackRequest
from ..dependencies import get_agent

router = APIRouter()

@router.post("/")
async def submit_feedback(request: FeedbackRequest):
    """Submit human correction to agent decision"""
    try:
        agent = get_agent()
        
        # Record feedback
        agent.provide_feedback(
            claim_id=request.claim_id,
            actual_outcome=request.actual_outcome or 
                          ('fraud' if request.human_decision in ['deny', 'deep'] else 'legitimate')
        )
        
        # Check if agent needs retraining
        performance = agent.get_performance_report()
        
        return {
            "status": "feedback_recorded",
            "claim_id": request.claim_id,
            "agent_decision": request.agent_decision,
            "human_decision": request.human_decision,
            "disagreement": request.agent_decision != request.human_decision,
            "current_accuracy": performance.get('accuracy', 'N/A'),
            "message": "Thank you for the feedback. The agent will learn from this."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def feedback_stats():
    """Get feedback statistics"""
    try:
        agent = get_agent()
        performance = agent.get_performance_report()
        
        return {
            "performance": performance,
            "total_feedback_received": len(agent.performance_tracker),
            "recent_decisions": len(agent.decision_history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))