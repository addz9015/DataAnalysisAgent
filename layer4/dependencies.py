# layer4/dependencies.py
"""Shared dependencies"""
import sys
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
        _pipeline = Layer1Pipeline()
        
    if _ensemble is None:
        print("Loading Layer 2 Ensemble...")
        _ensemble = StochasticEnsemble()
        # TODO: Load pre-trained weights here
        
    if _agent is None:
        print("Loading Layer 3 Agent...")
        import os
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