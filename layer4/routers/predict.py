"""Single prediction endpoint"""
from fastapi import APIRouter, HTTPException
import pandas as pd

from ..models.requests import PredictRequest
from ..models.responses import PredictResponse
from ..dependencies import get_pipeline, get_ensemble, get_agent

router = APIRouter()

@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Get fraud prediction for single claim"""
    try:
        # Get dependencies
        pipeline = get_pipeline()
        ensemble = get_ensemble()
        agent = get_agent()
        
        # Convert to DataFrame
        df = pd.DataFrame([request.dict()])
        
        # Layer 1: Process
        processed, _ = pipeline.process(df, export_csv=False)
        
        # Layer 2: Predict
        predictions = ensemble.predict(processed)
        
        # Layer 3: Agent decides
        result = agent.process_claim(predictions.iloc[0])
        
        # Format response
        decision = result['decision']
        explanations = result['explanations']
        
        return PredictResponse(
            claim_id=decision.claim_id,
            fraud_probability=result['analysis'].fraud_probability,
            agent_decision=decision.selected_action,
            confidence=decision.confidence,
            risk_score=decision.risk_score,
            requires_human_review=decision.requires_human_review,
            sla_hours=decision.sla_hours,
            investigation_depth=decision.investigation_depth,
            explanation=explanations.get('summary', 'No summary available'),
            explanation_source=explanations.get('source', 'unknown')
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))