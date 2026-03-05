"""Simple function to check fraud"""
import sys
sys.path.insert(0, '../..')

import pandas as pd
from layer1.core.pipeline import Layer1Pipeline
from layer2.core.ensemble import StochasticEnsemble
from layer3.core.agent_orchestrator import StochClaimAgent

# Global instances
_pipeline = None
_ensemble = None
_agent = None

def _init():
    """Lazy initialization"""
    global _pipeline, _ensemble, _agent
    if _pipeline is None:
        _pipeline = Layer1Pipeline()
        _ensemble = StochasticEnsemble()
        _agent = StochClaimAgent(use_llm=True, llm_threshold=0.6)
    return _pipeline, _ensemble, _agent

def is_fraud(**claim_data) -> dict:
    """
    One-line fraud check
    
    Usage:
        result = is_fraud(
            claim_id="C001",
            months_as_customer=6,
            age=28,
            total_claim_amount=25000,
            ...
        )
    """
    pipeline, ensemble, agent = _init()
    
    # Layer 1
    df = pd.DataFrame([claim_data])
    processed, _ = pipeline.process(df, export_csv=False)
    
    # Layer 2
    predictions = ensemble.predict(processed)
    
    # Layer 3
    result = agent.process_claim(predictions.iloc[0])
    
    decision = result['decision']
    analysis = result['analysis']
    
    return {
        'is_fraudulent': decision.risk_score > 70,
        'fraud_probability': float(analysis.fraud_probability),
        'decision': decision.selected_action,
        'confidence': float(decision.confidence),
        'risk_score': int(decision.risk_score),
        'explanation': result['explanations']['summary'],
        'requires_human_review': decision.requires_human_review
    }

def quick_check(claim_dict: dict) -> dict:
    """Wrapper that takes a dict"""
    return is_fraud(**claim_dict)