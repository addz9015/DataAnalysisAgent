# layer3/core/explanation_generator.py
"""
Generate human-readable explanations for agent decisions
"""

import pandas as pd
from typing import Dict, List
import json

class ExplanationGenerator:
    """
    Create explanations for decisions (template-based or LLM)
    """
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        
    def generate(self, decision: 'AgentDecision', 
                 analysis: 'SituationAnalysis') -> Dict[str, str]:
        """
        Generate multi-format explanations
        """
        explanations = {
            'summary': self._summary_explanation(decision, analysis),
            'detailed': self._detailed_explanation(decision, analysis),
            'technical': self._technical_explanation(decision, analysis),
            'json': self._structured_explanation(decision, analysis)
        }
        
        if self.use_llm:
            explanations['natural_language'] = self._llm_explanation(decision, analysis)
        
        return explanations
    
    def _summary_explanation(self, decision, analysis) -> str:
        """One-line summary"""
        action_desc = {
            'approve': 'automatically approved',
            'fast_track': 'fast-tracked with light review',
            'standard': 'sent for standard investigation',
            'deep': 'flagged for deep investigation',
            'deny': 'automatically denied'
        }
        
        return (
            f"Claim {decision.claim_id} {action_desc.get(decision.selected_action, 'processed')} "
            f"with {decision.confidence:.0%} confidence. "
            f"Risk score: {decision.risk_score}/100."
        )
    
    def _detailed_explanation(self, decision, analysis) -> str:
        """Paragraph explanation for business users"""
        lines = [
            f"DECISION: {decision.selected_action.upper()}",
            "",
            f"This claim has been {decision.selected_action} based on the following analysis:",
            "",
            f"• Fraud Probability: {analysis.fraud_probability:.1%} (confidence: {decision.confidence:.0%})",
            f"• Expected Fraud Loss: ${analysis.expected_fraud_loss:,.2f}",
            f"• Investigation Cost: ${analysis.investigation_cost[decision.selected_action]:,.2f}",
            f"• Time Pressure: {analysis.time_pressure}",
            "",
            "Key Evidence:",
            f"  - Red Flags: {analysis.key_evidence['red_flags']}/4",
            f"  - Incident Severity: {analysis.key_evidence['severity']}",
            f"  - Claim-to-Premium Ratio: {analysis.key_evidence['claim_premium_ratio']}x",
            "",
            f"Reasoning: {decision.reasoning}",
            ""
        ]
        
        if decision.requires_human_review:
            lines.append("⚠️  This claim requires human review due to high uncertainty or risk.")
        
        if decision.alternative_actions:
            lines.extend([
                "",
                "Alternative Actions Considered:",
                f"  1. {decision.alternative_actions[0]['action']}: "
                f"EV=${decision.alternative_actions[0]['expected_value']}",
                f"  2. {decision.alternative_actions[1]['action']}: "
                f"EV=${decision.alternative_actions[1]['expected_value']}" if len(decision.alternative_actions) > 1 else ""
            ])
        
        return "\n".join(lines)
    
    def _technical_explanation(self, decision, analysis) -> str:
        """Technical details for data scientists"""
        return json.dumps({
            'decision': decision.selected_action,
            'confidence': decision.confidence,
            'fraud_probability': analysis.fraud_probability,
            'uncertainty': analysis.uncertainty,
            'expected_value_by_action': analysis.investigation_cost,
            'risk_score': decision.risk_score,
            'sla_hours': decision.sla_hours,
            'requires_human_review': decision.requires_human_review,
            'investigation_depth': decision.investigation_depth
        }, indent=2)
    
    def _structured_explanation(self, decision, analysis) -> Dict:
        """Structured dict for API/frontend"""
        return {
            'claim_id': decision.claim_id,
            'decision': {
                'action': decision.selected_action,
                'confidence': decision.confidence,
                'risk_score': decision.risk_score
            },
            'rationale': {
                'fraud_probability': analysis.fraud_probability,
                'key_factors': analysis.key_evidence,
                'reasoning_text': decision.reasoning
            },
            'next_steps': {
                'sla_hours': decision.sla_hours,
                'investigation_depth': decision.investigation_depth,
                'human_review_required': decision.requires_human_review
            }
        }
    
    def _llm_explanation(self, decision, analysis) -> str:
        """Natural language explanation using LLM"""
        try:
            from layer3.llm.explainer_llm import LLMExplainer
            explainer = LLMExplainer()
            
            # Format inputs for LLMExplainer
            claim_id = str(decision.claim_id)
            decision_action = str(decision.selected_action)
            fraud_prob = float(analysis.fraud_probability)
            confidence = float(decision.confidence)
            key_evidence = analysis.key_evidence
            reasoning = str(decision.reasoning)
            
            return explainer.explain_decision(
                claim_id=claim_id,
                decision=decision_action,
                fraud_probability=fraud_prob,
                confidence=confidence,
                key_evidence=key_evidence,
                reasoning=reasoning
            )
        except Exception as e:
            return f"[LLM Error] Failed to generate explanation: {e}"