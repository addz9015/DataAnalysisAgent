# layer4/dependencies.py
"""Shared dependencies"""
import os
import sys
import pandas as pd
sys.path.insert(0, '..')

from functools import lru_cache
from fastapi import HTTPException, Header

from layer1.core.pipeline import Layer1Pipeline
from layer2.core.ensemble import StochasticEnsemble
from layer3.core.agent_orchestrator import StochClaimAgent
from .config import settings

# Singleton instances
_pipeline = None
_ensemble = None
_agent = None

def _init_models():
    """Initialize all models once"""
    global _pipeline, _ensemble, _agent
    
    if _pipeline is None:
        print("Loading Layer 1 Pipeline...")
        _pipeline = Layer1Pipeline(config={"storage_path": os.path.join("data", "processed")})

    if _ensemble is None:
        print("Loading Layer 2 Ensemble...")
        _ensemble = StochasticEnsemble()

        bootstrap_path = os.path.join("data", "processed", "processed_features.csv")
        if os.path.exists(bootstrap_path):
            bootstrap_df = pd.read_csv(bootstrap_path)

            # Calibrate Layer 1 scaling for single-claim inference using reference data.
            if _pipeline is not None and hasattr(_pipeline, "preprocessor"):
                _pipeline.preprocessor.fit_scaler_reference(bootstrap_df)

            hmm_features = [
                "age_scaled",
                "total_claim_amount_scaled",
                "claim_to_premium_ratio_scaled",
                "injury_ratio_scaled",
                "property_ratio_scaled",
                "vehicle_ratio_scaled",
                "severity_score_scaled",
                "complexity_score_scaled",
                "red_flag_count",
            ]
            available_hmm_features = [f for f in hmm_features if f in bootstrap_df.columns]
            if len(available_hmm_features) >= 3:
                _ensemble.fit(bootstrap_df, hmm_features=available_hmm_features)
                print(f"Layer 2 Ensemble bootstrapped from {bootstrap_path}")
            else:
                print("Warning: insufficient features to bootstrap Layer 2 Ensemble")
        else:
            print(f"Warning: bootstrap file missing at {bootstrap_path}")
        
    if _agent is None:
        print("Loading Layer 3 Agent...")
        _agent = StochClaimAgent(
            use_llm=bool(os.getenv("GROQ_API_KEY")),
            llm_provider='groq',
            llm_threshold=settings.LLM_THRESHOLD
        )
    
    return _pipeline, _ensemble, _agent

@lru_cache()
def get_pipeline():
    """Get or create Layer 1 pipeline"""
    pipeline, _, _ = _init_models()
    return pipeline

@lru_cache()
def get_ensemble():
    """Get or create Layer 2 ensemble"""
    _, ensemble, _ = _init_models()
    return ensemble

@lru_cache()
def get_agent():
    """Get or create Layer 3 agent"""
    _, _, agent = _init_models()
    return agent

def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key"""
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

def get_models():
    """Get all models at once (for batch processing)"""
    return _init_models()