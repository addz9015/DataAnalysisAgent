"""Request schemas"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ClaimData(BaseModel):
    claim_id: str
    months_as_customer: int = Field(..., ge=0, le=600)
    age: int = Field(..., ge=16, le=100)
    policy_annual_premium: float = Field(..., gt=0)
    incident_severity: str
    total_claim_amount: float = Field(..., ge=0)
    injury_claim: float = Field(..., ge=0)
    property_claim: float = Field(..., ge=0)
    vehicle_claim: float = Field(..., ge=0)
    incident_type: str
    collision_type: str
    authorities_contacted: str
    witnesses: int = Field(..., ge=0)
    witness_present: Optional[str] = "UNKNOWN"
    police_report_available: Optional[str] = "UNKNOWN"

class PredictRequest(ClaimData):
    pass

class BatchRequest(BaseModel):
    claims: List[ClaimData]

class QueryRequest(BaseModel):
    question: str
    claim_data: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    claim_id: str
    agent_decision: str
    human_decision: str
    reason: Optional[str] = None
    actual_outcome: Optional[str] = None