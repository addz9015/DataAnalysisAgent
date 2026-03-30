"""Single prediction endpoint"""
from fastapi import APIRouter, HTTPException
import pandas as pd
import uuid
import logging

from ..models.requests import PredictRequest
from ..models.responses import PredictResponse
from ..dependencies import get_pipeline, get_ensemble, get_agent
from shared.claim_database import upsert_claim_prediction

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

@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Get fraud prediction for single claim"""
    try:
        # Get dependencies
        pipeline = get_pipeline()
        ensemble = get_ensemble()
        agent = get_agent()
        
        # Convert to DataFrame
        payload = request.model_dump()
        if not payload.get('claim_id'):
            payload['claim_id'] = f"ID_{uuid.uuid4().hex[:8]}"
        payload['witnesses'] = _resolve_witness_count(payload)

        df = pd.DataFrame([payload])
        
        # Layer 1: Process
        processed, _ = pipeline.process(df, export_csv=False)
        
        # Layer 2: Predict
        predictions = ensemble.predict(processed)
        
        # Layer 3: Agent decides
        result = agent.process_claim(predictions.iloc[0])
        
        # Format response
        decision = result['decision']
        explanations = result['explanations']
        
        response_payload = {
            "claim_id": decision.claim_id,
            "fraud_probability": result['analysis'].fraud_probability,
            "agent_decision": decision.selected_action,
            "confidence": decision.confidence,
            "risk_score": decision.risk_score,
            "requires_human_review": decision.requires_human_review,
            "sla_hours": decision.sla_hours,
            "investigation_depth": decision.investigation_depth,
            "explanation": explanations.get('summary', 'No summary available'),
            "explanation_source": explanations.get('source', 'unknown'),
            "human_review_note": result.get('human_review_note'),
        }
        response = PredictResponse(**response_payload)

        try:
            upsert_claim_prediction(
                input_claim=payload,
                prediction=response_payload,
                source_type="single_entry",
            )
        except Exception:
            # Persistence should not block the API result.
            logger.warning("Failed to persist single prediction for claim %s", response_payload.get("claim_id"), exc_info=True)

        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))