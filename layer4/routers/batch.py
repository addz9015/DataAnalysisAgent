"""Batch prediction endpoint"""
from fastapi import APIRouter, HTTPException
import pandas as pd

from ..models.requests import BatchRequest
from ..models.responses import BatchResponse, PredictResponse
from ..dependencies import get_pipeline, get_ensemble, get_agent

router = APIRouter()

@router.post("/", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """Process multiple claims"""
    try:
        pipeline = get_pipeline()
        ensemble = get_ensemble()
        agent = get_agent()
        
        # Convert to DataFrame
        claims_data = [c.dict() for c in request.claims]
        df = pd.DataFrame(claims_data)
        
        # Process all layers
        processed, _ = pipeline.process(df, export_csv=False)
        predictions = ensemble.predict(processed)
        results_df = agent.process_batch(predictions, show_progress=False)
        
        # Format responses
        responses = []
        for _, row in results_df.iterrows():
            responses.append(PredictResponse(
                claim_id=row['claim_id'],
                fraud_probability=row['fraud_probability'],
                agent_decision=row['agent_action'],
                confidence=row['agent_confidence'],
                risk_score=row['risk_score'],
                requires_human_review=row['requires_human_review'],
                sla_hours=row['sla_hours'],
                investigation_depth=row['investigation_depth'],
                explanation=row.get('explanation_summary', 'No summary available'),
                explanation_source=row.get('explanation_source', 'unknown')
            ))
        
        # Summary
        summary = {
            'high_risk': len([r for r in responses if r.risk_score > 70]),
            'human_review_needed': len([r for r in responses if r.requires_human_review]),
            'avg_fraud_probability': sum(r.fraud_probability for r in responses) / len(responses),
            'decision_distribution': {}
        }
        
        for decision in ['approve', 'fast_track', 'standard', 'deep', 'deny']:
            count = len([r for r in responses if r.agent_decision == decision])
            summary['decision_distribution'][decision] = count
        
        return BatchResponse(
            total_processed=len(responses),
            results=responses,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))