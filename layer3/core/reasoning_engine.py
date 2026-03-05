# layer3/core/reasoning_engine.py
"""
Reasoning engine: Analyze situations and evaluate options
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("layer3.reasoning")

@dataclass
class SituationAnalysis:
    """Structured analysis of a claim situation"""
    claim_id: str
    fraud_probability: float
    uncertainty: float  # Confidence interval width
    expected_fraud_loss: float
    investigation_cost: Dict[str, float]
    time_pressure: str  # 'low', 'medium', 'high'
    risk_tolerance: str  # 'conservative', 'balanced', 'aggressive'
    key_evidence: Dict[str, any]

class ReasoningEngine:
    """
    Analyze claim situations using stochastic model outputs
    """
    
    INVESTIGATION_COSTS = {
        'approve': 0,
        'fast_track': 50,
        'standard': 200,
        'deep': 1000,
        'deny': 100
    }
    
    def __init__(self, risk_tolerance: str = 'balanced'):
        self.risk_tolerance = risk_tolerance
        
    def analyze(self, claim_row: pd.Series) -> SituationAnalysis:
        """
        Deep analysis of single claim situation
        """
        # Calculate uncertainty from model agreement
        hmm_prob = claim_row['fraud_probability']
        gr_prob = claim_row.get('P_fraud_detected_gr', hmm_prob)
        
        # Uncertainty = disagreement between models
        uncertainty = abs(hmm_prob - gr_prob)
        
        # Expected fraud loss
        claim_amount = claim_row['total_claim_amount']
        expected_fraud_loss = hmm_prob * claim_amount * 0.8  # 80% loss if fraud
        
        # Time pressure based on claim age
        months = claim_row.get('months_as_customer', 0)
        if months < 6:
            time_pressure = 'high'  # New customer, fast resolution expected
        elif months < 24:
            time_pressure = 'medium'
        else:
            time_pressure = 'low'  # Long-term customer, can take time
        
        # Investigation costs with uncertainty adjustment
        investigation_cost = {
            action: cost * (1 + uncertainty)  # Higher uncertainty = higher effective cost
            for action, cost in self.INVESTIGATION_COSTS.items()
        }
        
        # Key evidence for reasoning
        key_evidence = {
            'red_flags': int(claim_row['red_flag_count']),
            'severity': claim_row['incident_severity'],
            'claim_premium_ratio': round(claim_row['claim_to_premium_ratio'], 2),
            'witness_present': claim_row['witness_present'],
            'police_report': claim_row['police_report_available']
        }
        
        return SituationAnalysis(
            claim_id=str(claim_row.get('claim_id', 'unknown')),
            fraud_probability=float(hmm_prob),
            uncertainty=float(uncertainty),
            expected_fraud_loss=float(expected_fraud_loss),
            investigation_cost=investigation_cost,
            time_pressure=time_pressure,
            risk_tolerance=self.risk_tolerance,
            key_evidence=key_evidence
        )
    
    def evaluate_options(self, analysis: SituationAnalysis) -> List[Dict]:
        """
        Evaluate all possible actions for this situation
        """
        options = []
        
        for action, cost in analysis.investigation_cost.items():
            # Calculate net expected value
            if action == 'approve':
                # Risk: fraud loss. Benefit: customer satisfaction
                expected_value = -analysis.expected_fraud_loss + 100  # Satisfaction bonus
                risk_level = 'high' if analysis.fraud_probability > 0.5 else 'low'
                
            elif action == 'deny':
                # Risk: false positive. Benefit: prevent fraud
                false_positive_cost = (1 - analysis.fraud_probability) * 500
                expected_value = analysis.expected_fraud_loss * 0.5 - false_positive_cost
                risk_level = 'high' if analysis.fraud_probability < 0.3 else 'low'
                
            elif action == 'fast_track':
                # Light check, quick resolution
                fraud_prevention = analysis.fraud_probability * 0.1 * analysis.expected_fraud_loss
                expected_value = fraud_prevention - cost
                risk_level = 'medium'
                
            elif action == 'standard':
                # Moderate investigation
                fraud_prevention = analysis.fraud_probability * 0.3 * analysis.expected_fraud_loss
                expected_value = fraud_prevention - cost
                risk_level = 'low'
                
            else:  # deep
                # Thorough investigation
                fraud_prevention = analysis.fraud_probability * 0.6 * analysis.expected_fraud_loss
                expected_value = fraud_prevention - cost
                risk_level = 'very_low'
            
            # Adjust for time pressure
            if analysis.time_pressure == 'high' and action in ['standard', 'deep']:
                expected_value -= 200  # Penalty for slow actions
            
            # Adjust for risk tolerance
            if self.risk_tolerance == 'conservative' and risk_level in ['high', 'medium']:
                expected_value -= 100
            elif self.risk_tolerance == 'aggressive' and risk_level in ['very_low', 'low']:
                expected_value += 50
            
            options.append({
                'action': action,
                'expected_value': round(expected_value, 2),
                'cost': round(cost, 2),
                'risk_level': risk_level,
                'fraud_prevention_potential': round(analysis.fraud_probability * 
                                                   (0.1 if action == 'fast_track' else
                                                    0.3 if action == 'standard' else
                                                    0.6 if action == 'deep' else 0), 2)
            })
        
        # Sort by expected value
        options.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return options
    
    def detect_anomalies(self, claim_row: pd.Series, 
                        historical_stats: Dict) -> List[str]:
        """
        Detect if claim is anomalous compared to historical patterns
        """
        anomalies = []
        
        # Check claim amount
        if claim_row['total_claim_amount'] > historical_stats.get('amount_99th', float('inf')):
            anomalies.append("Claim amount in top 1% historically")
        
        # Check ratio
        if claim_row['claim_to_premium_ratio'] > 20:
            anomalies.append("Claim-to-premium ratio extremely high (>20x)")
        
        # Check red flags
        if claim_row['red_flag_count'] >= 4:
            anomalies.append("Maximum red flag count - highly suspicious pattern")
        
        # Check speed of claim (if we had incident date)
        # if days_between_incident_and_claim < 1:
        #     anomalies.append("Claim filed same day as incident")
        
        return anomalies