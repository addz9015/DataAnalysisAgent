"""
Use templates for most, LLM only for complex cases
"""

class HybridExplainer:
    """
    Save API costs: Template for simple, LLM for complex
    """
    
    TEMPLATES = {
        'approve': "Low risk claim ({fraud_prob:.0%} fraud probability). No anomalies detected. Approved for fast processing.",
        'fast_track': "Moderate risk ({fraud_prob:.0%}). {red_flags} red flags noted. Quick review recommended.",
        'standard': "Elevated risk ({fraud_prob:.0%}). Multiple indicators: {indicators}. Standard investigation.",
        'deep': "High fraud risk ({fraud_prob:.0%}). Critical: {indicators}. Thorough investigation required.",
        'deny': "Fraud probability {fraud_prob:.0%} exceeds threshold. Claim denied."
    }
    
    def __init__(self, llm_explainer, llm_threshold: float = 0.6):
        self.llm = llm_explainer
        self.llm_threshold = llm_threshold  # Only use LLM for fraud_prob > 60%
        self.last_source = 'template'
        
    def explain(self, claim_data: dict) -> str:
        """
        Choose template or LLM based on complexity
        """
        fraud_prob = claim_data.get('fraud_probability', 0)
        decision = claim_data.get('agent_decision', 'standard')
        
        # Simple cases: use template
        if fraud_prob < self.llm_threshold:
            self.last_source = 'template'
            return self._template_explain(claim_data)
        
        # Complex cases: use LLM
        explanation = self.llm.explain_decision(
            claim_id=claim_data.get('claim_id', 'unknown'),
            decision=decision,
            fraud_probability=fraud_prob,
            confidence=claim_data.get('confidence', 0.0),
            key_evidence={
                'red_flags': claim_data.get('red_flag_count', 0),
                'claim_premium_ratio': claim_data.get('claim_to_premium_ratio', 0),
                'severity': claim_data.get('severity', 'Unknown')
            },
            reasoning=claim_data.get('reasoning', '')
        )
        llm_source = getattr(self.llm, 'last_source', 'llm')
        self.last_source = 'llm' if llm_source == 'llm' else 'template'
        return explanation
    
    def _template_explain(self, data: dict) -> str:
        """Generate from template"""
        template = self.TEMPLATES.get(data['agent_decision'], self.TEMPLATES['standard'])
        
        indicators = []
        if data.get('red_flag_count', 0) > 0:
            indicators.append(f"{data['red_flag_count']} red flags")
        if data.get('claim_to_premium_ratio', 0) > 10:
            indicators.append("high claim ratio")
        
        return template.format(
            fraud_prob=data.get('fraud_probability', 0),
            red_flags=data.get('red_flag_count', 0),
            indicators=", ".join(indicators) if indicators else "standard review"
        )