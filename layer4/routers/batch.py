"""Batch prediction endpoint"""
from fastapi import APIRouter, HTTPException
import pandas as pd
import uuid
import logging

from ..models.requests import BatchRequest
from ..models.responses import BatchResponse, PredictResponse
from ..dependencies import get_pipeline, get_ensemble, get_agent
from shared.claim_database import upsert_claim_predictions

router = APIRouter()
logger = logging.getLogger("layer4.api")


def _resolve_witness_count(payload: dict) -> int:
    """Normalize witness count from optional numeric and presence flag fields."""
    raw = payload.get("witnesses")

    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            raw = None

    if raw is None:
        witness_present = str(payload.get("witness_present", "")).strip().lower()
        return 1 if witness_present in {"yes", "y", "1", "true"} else 0

    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        witness_present = str(payload.get("witness_present", "")).strip().lower()
        return 1 if witness_present in {"yes", "y", "1", "true"} else 0

@router.post("/", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """Process multiple claims"""
    try:
        if not request.claims:
            raise HTTPException(status_code=422, detail="At least one claim is required")

        pipeline = get_pipeline()
        ensemble = get_ensemble()
        agent = get_agent()
        
        # Convert to DataFrame
        claims_data = []
        for claim in request.claims:
            payload = claim.model_dump()
            if not payload.get('claim_id'):
                payload['claim_id'] = f"ID_{uuid.uuid4().hex[:8]}"
            payload['witnesses'] = _resolve_witness_count(payload)
            claims_data.append(payload)
        df = pd.DataFrame(claims_data)
        
        # Process all layers
        processed, _ = pipeline.process(df, export_csv=False)
        predictions = ensemble.predict(processed)
        results_df = agent.process_batch(predictions, show_progress=False)
        
        # Format responses
        responses = []
        persistence_entries = []
        for idx, (_, row) in enumerate(results_df.iterrows()):
            response_payload = {
                "claim_id": row['claim_id'],
                "fraud_probability": row['fraud_probability'],
                "agent_decision": row['agent_action'],
                "confidence": row['agent_confidence'],
                "risk_score": row['risk_score'],
                "requires_human_review": row['requires_human_review'],
                "sla_hours": row['sla_hours'],
                "investigation_depth": row['investigation_depth'],
                "explanation": row.get('explanation_summary', 'No summary available'),
                "explanation_source": row.get('explanation_source', 'unknown'),
                "human_review_note": row.get('human_review_note'),
            }
            responses.append(PredictResponse(**response_payload))
            if idx < len(claims_data):
                persistence_entries.append((claims_data[idx], response_payload))

        if persistence_entries:
            try:
                upsert_claim_predictions(
                    entries=persistence_entries,
                    source_type="batch_upload",
                )
            except Exception:
                # Persistence should not block batch API responses.
                logger.warning("Failed to persist one or more batch prediction rows", exc_info=True)
        
        # Summary
        summary = {
            'high_risk': len([r for r in responses if r.risk_score > 70]),
            'human_review_needed': len([r for r in responses if r.requires_human_review]),
            'avg_fraud_probability': (
                sum(r.fraud_probability for r in responses) / len(responses)
                if responses else 0.0
            ),
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))