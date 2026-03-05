"""Natural language query endpoint"""
from fastapi import APIRouter, HTTPException

from ..models.requests import QueryRequest
from ..dependencies import get_agent, get_pipeline, get_ensemble
import pandas as pd

router = APIRouter()

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """Ask agent a natural language question"""
    try:
        # Simple intent detection
        question = request.question.lower()
        
        # Fraud check intent
        if any(w in question for w in ["fraud", "fraudulent", "scam", "fake", "risk"]):
            if not request.claim_data:
                return {
                    "answer": "Please provide claim data to check for fraud.",
                    "intent": "fraud_check"
                }
            
            # Process claim
            pipeline = get_pipeline()
            ensemble = get_ensemble()
            agent = get_agent()
            
            df = pd.DataFrame([request.claim_data])
            processed, _ = pipeline.process(df, export_csv=False)
            predictions = ensemble.predict(processed)
            result = agent.process_claim(predictions.iloc[0])
            
            analysis = result['analysis']
            decision = result['decision']
            
            answer = (
                f"This claim has a {analysis.fraud_probability:.1%} probability of fraud. "
                f"My recommendation is to {decision.selected_action} this claim. "
                f"Risk score: {decision.risk_score}/100."
            )
            
            return {
                "question": request.question,
                "answer": answer,
                "intent": "fraud_check",
                "fraud_probability": analysis.fraud_probability,
                "recommended_action": decision.selected_action
            }
        
        # Explanation intent
        elif any(w in question for w in ["why", "explain", "reason", "how"]):
            return {
                "answer": "I analyze multiple factors including claim amount, customer history, incident details, and red flags to make decisions.",
                "intent": "explanation"
            }
        
        # Help intent
        elif any(w in question for w in ["help", "what can you do", "capabilities"]):
            return {
                "answer": "I can: 1) Check if claims are fraudulent, 2) Recommend investigation actions, 3) Explain my decisions, 4) Process claims in batch.",
                "intent": "help"
            }
        
        # Default
        else:
            return {
                "answer": "I can help check claims for fraud risk. Please provide claim details or ask a specific question.",
                "intent": "unknown"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quick-check")
async def quick_check(claim_data: dict):
    """Quick yes/no fraud check"""
    try:
        pipeline = get_pipeline()
        ensemble = get_ensemble()
        agent = get_agent()
        
        df = pd.DataFrame([claim_data])
        processed, _ = pipeline.process(df, export_csv=False)
        predictions = ensemble.predict(processed)
        result = agent.process_claim(predictions.iloc[0])
        
        fraud_prob = result['analysis'].fraud_probability
        
        return {
            "is_fraudulent": fraud_prob > 0.7,
            "fraud_probability": fraud_prob,
            "risk_level": "high" if fraud_prob > 0.7 else "medium" if fraud_prob > 0.3 else "low",
            "recommended_action": result['decision'].selected_action
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))