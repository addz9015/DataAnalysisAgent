"""Response schemas"""
from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictResponse(BaseModel):
    claim_id: str
    fraud_probability: float
    agent_decision: str
    confidence: float
    risk_score: int
    requires_human_review: bool
    sla_hours: int
    investigation_depth: Optional[int]
    explanation: str
    explanation_source: str

class BatchResponse(BaseModel):
    total_processed: int
    results: List[PredictResponse]
    summary: Dict

class ExplainResponse(BaseModel):
    claim_id: str
    summary: str
    detailed: str
    technical: str

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool