# layer3/core/action_selector.py
"""
Action Selector: The core agent that makes autonomous decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("layer3.agent")

class ActionType(Enum):
    APPROVE = "approve"
    FAST_TRACK = "fast_track"
    STANDARD = "standard"
    DEEP = "deep"
    DENY = "deny"

@dataclass
class AgentDecision:
    """Structured decision output"""
    claim_id: str
    selected_action: str
    confidence: float
    reasoning: str
    alternative_actions: List[Dict]
    investigation_depth: Optional[int]  # For investigation actions
    sla_hours: int
    requires_human_review: bool
    risk_score: float
    sanity_override: bool = False

class ActionSelector:
    """
    Autonomous agent that selects optimal actions
    """
    
    def __init__(self, 
                 auto_approve_threshold: float = 0.15,
                 auto_deny_threshold: float = 0.98,
                 human_review_threshold: float = 0.60):
        
        self.auto_approve_threshold = auto_approve_threshold
        self.auto_deny_threshold = auto_deny_threshold
        self.human_review_threshold = human_review_threshold
        
        # SLA by action type (hours)
        self.sla_map = {
            'approve': 24,
            'fast_track': 48,
            'standard': 120,
            'deep': 240,
            'deny': 72
        }
        
    def decide(self, 
               analysis: 'SituationAnalysis',
               evaluated_options: List[Dict],
               anomalies: List[str]) -> AgentDecision:
        """
        Make autonomous decision on claim
        """
        fraud_prob = analysis.fraud_probability
        uncertainty = analysis.uncertainty
        sanity_override = False
        key = analysis.key_evidence

        red_flags = int(key.get('red_flags', 0) or 0)
        claim_premium_ratio = float(key.get('claim_premium_ratio', 999) or 999)
        severity = str(key.get('severity', '')).strip().lower()
        witness_present = str(key.get('witness_present', '')).strip().lower()
        police_report = str(key.get('police_report', '')).strip().lower()
        
        # Calculate confidence (inverse of uncertainty, normalized)
        confidence = max(0, 1 - uncertainty * 2)
        
        # Determine if human review needed
        requires_human = (
            fraud_prob > self.human_review_threshold or
            len(anomalies) >= 2 or
            uncertainty > 0.3 or
            analysis.time_pressure == 'high' and fraud_prob > 0.4
        )

        # Sanity override: documented, low-impact claims should not be denied solely by model score.
        low_risk_documentation = (
            witness_present in {'yes', 'y', '1', 'true'} and
            police_report in {'yes', 'y', '1', 'true'}
        )
        low_impact_severity = severity in {'trivial damage', 'minor damage'}
        low_financial_ratio = claim_premium_ratio <= 3.0

        if (
            red_flags == 0 and
            low_risk_documentation and
            low_impact_severity and
            low_financial_ratio and
            fraud_prob < 0.95 and
            not anomalies
        ):
            selected = 'approve' if claim_premium_ratio <= 2.0 else 'fast_track'
            reasoning = (
                "Low-risk sanity override applied: fully documented low-impact claim with no red flags "
                "and a moderate claim-to-premium ratio."
            )
            sanity_override = True
        else:
            # Select action based on thresholds and optimization
            if fraud_prob < self.auto_approve_threshold and not anomalies:
                # Very low risk - auto approve
                selected = 'approve'
                reasoning = f"Fraud probability {fraud_prob:.1%} below threshold. No anomalies detected."
                
            elif fraud_prob > self.auto_deny_threshold:
                # Very high risk - auto deny
                selected = 'deny'
                reasoning = f"Fraud probability {fraud_prob:.1%} exceeds auto-deny threshold."
                
            else:
                # Use optimization result from reasoning engine
                best_option = evaluated_options[0]
                selected = best_option['action']
                
                reasoning = (
                    f"Selected {selected} based on expected value ${best_option['expected_value']}. "
                    f"Prevents estimated ${best_option['fraud_prevention_potential']:.0f} in fraud losses "
                    f"at cost of ${best_option['cost']:.0f}."
                )
        
        # Determine investigation depth if applicable
        investigation_depth = None
        if selected in ['standard', 'deep']:
            # Use Gambler's Ruin to determine optimal depth
            initial_evidence = analysis.key_evidence['red_flags']
            # Map to 1-5 depth scale
            investigation_depth = min(5, max(1, initial_evidence + 1))
        
        # Calculate risk score (0-100)
        risk_score = min(100, int(fraud_prob * 100 + len(anomalies) * 10))

        if sanity_override:
            requires_human = False
            risk_score = min(risk_score, 20)
        
        # Get alternative actions (top 3 excluding selected)
        alternative_actions = [opt for opt in evaluated_options if opt['action'] != selected][:3]
        
        return AgentDecision(
            claim_id=analysis.claim_id,
            selected_action=selected,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            alternative_actions=alternative_actions,
            investigation_depth=investigation_depth,
            sla_hours=self.sla_map[selected],
            requires_human_review=requires_human,
            risk_score=risk_score,
            sanity_override=sanity_override,
        )
    
    def batch_decide(self, 
                     analyses: List['SituationAnalysis'],
                     options_list: List[List[Dict]],
                     anomalies_list: List[List[str]]) -> List[AgentDecision]:
        """Process multiple claims"""
        return [
            self.decide(analysis, options, anomalies)
            for analysis, options, anomalies in 
            zip(analyses, options_list, anomalies_list)
        ]
    
    def adapt_thresholds(self, performance_history: pd.DataFrame):
        """
        Adapt thresholds based on past performance (online learning)
        """
        if len(performance_history) < 100:
            return  # Not enough data
        
        # Calculate false positive/negative rates
        false_positives = len(performance_history[
            (performance_history['decision'] == 'deny') &
            (performance_history['actual_fraud'] == False)
        ])
        false_negatives = len(performance_history[
            (performance_history['decision'] == 'approve') &
            (performance_history['actual_fraud'] == True)
        ])
        
        fp_rate = false_positives / len(performance_history)
        fn_rate = false_negatives / len(performance_history)
        
        # Adjust thresholds
        if fp_rate > 0.05:  # Too many false positives
            previous = self.auto_deny_threshold
            # Keep auto-deny threshold monotonic when tuning online.
            self.auto_deny_threshold = max(
                previous,
                min(0.995, previous + 0.05)
            )
            logger.info(f"Raised auto-deny threshold to {self.auto_deny_threshold}")
        
        if fn_rate > 0.02:  # Too many false negatives
            self.auto_approve_threshold = max(0.05, self.auto_approve_threshold - 0.05)
            logger.info(f"Lowered auto-approve threshold to {self.auto_approve_threshold}")