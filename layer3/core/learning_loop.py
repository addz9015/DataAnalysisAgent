# layer3/core/learning_loop.py
"""
Online learning: Adapt agent based on outcomes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger("layer3.learning")

class LearningLoop:
    """
    Continuous learning from claim outcomes
    """
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.outcomes = deque(maxlen=memory_size)
        self.model_updates = []
        
    def record_outcome(self, 
                       claim_id: str,
                       predicted_action: str,
                       predicted_fraud_prob: float,
                       actual_outcome: str,  # 'fraud', 'legitimate', 'unknown'
                       actual_cost: Optional[float] = None):
        """
        Record actual outcome for learning
        """
        outcome = {
            'claim_id': claim_id,
            'timestamp': pd.Timestamp.now(),
            'predicted_action': predicted_action,
            'predicted_fraud_prob': predicted_fraud_prob,
            'actual_outcome': actual_outcome,
            'actual_cost': actual_cost,
            'correct': self._evaluate_correctness(predicted_action, actual_outcome)
        }
        
        self.outcomes.append(outcome)
        logger.info(f"Recorded outcome for {claim_id}: {actual_outcome}")
        
        return outcome
    
    def _evaluate_correctness(self, action: str, outcome: str) -> bool:
        """Check if decision was correct"""
        if outcome == 'fraud':
            return action in ['deny', 'deep', 'standard']
        elif outcome == 'legitimate':
            return action in ['approve', 'fast_track']
        return False  # unknown
    
    def calculate_regret(self) -> Dict:
        """
        Calculate opportunity cost of suboptimal decisions
        """
        if len(self.outcomes) < 10:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.outcomes)
        
        # False positives: Denied legitimate claims
        false_positives = df[(df['predicted_action'] == 'deny') & 
                            (df['actual_outcome'] == 'legitimate')]
        
        # False negatives: Approved fraudulent claims
        false_negatives = df[(df['predicted_action'].isin(['approve', 'fast_track'])) & 
                            (df['actual_outcome'] == 'fraud')]
        
        fp_cost = len(false_positives) * 500  # Customer churn cost
        fn_cost = len(false_negatives) * 8000  # Average fraud loss
        
        return {
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'fp_rate': len(false_positives) / len(df),
            'fn_rate': len(false_negatives) / len(df),
            'estimated_fp_cost': fp_cost,
            'estimated_fn_cost': fn_cost,
            'total_regret': fp_cost + fn_cost
        }
    
    def suggest_threshold_adjustments(self) -> Dict:
        """Suggest new thresholds based on performance"""
        regret = self.calculate_regret()
        
        if regret.get('status') == 'insufficient_data':
            return regret
        
        suggestions = {}
        
        # If too many false positives, raise bar for denial
        if regret['fp_rate'] > 0.05:
            suggestions['auto_deny_threshold'] = 'increase by 0.05'
        
        # If too many false negatives, lower bar for approval
        if regret['fn_rate'] > 0.02:
            suggestions['auto_approve_threshold'] = 'decrease by 0.05'
        
        # If investigation costs high, suggest more auto-decisions
        investigation_rate = sum(
            1 for o in self.outcomes 
            if o['predicted_action'] in ['standard', 'deep']
        ) / len(self.outcomes)
        
        if investigation_rate > 0.5:
            suggestions['investigation_strategy'] = 'too_many_investigations'
        
        return {
            'current_regret': regret,
            'suggested_adjustments': suggestions,
            'confidence': min(1.0, len(self.outcomes) / 500)
        }
    
    def update_strategy(self, agent) -> bool:
        """
        Actually update agent strategy based on learning
        """
        suggestions = self.suggest_threshold_adjustments()
        
        if 'suggested_adjustments' not in suggestions:
            return False
        
        adjustments = suggestions['suggested_adjustments']
        updated = False
        
        if 'auto_deny_threshold' in adjustments:
            agent.selector.auto_deny_threshold = min(
                0.95, 
                agent.selector.auto_deny_threshold + 0.05
            )
            updated = True
            logger.info("Updated auto_deny_threshold based on learning")
        
        if 'auto_approve_threshold' in adjustments:
            agent.selector.auto_approve_threshold = max(
                0.05,
                agent.selector.auto_approve_threshold - 0.05
            )
            updated = True
            logger.info("Updated auto_approve_threshold based on learning")
        
        if updated:
            self.model_updates.append({
                'timestamp': pd.Timestamp.now(),
                'adjustments': adjustments,
                'outcomes_count': len(self.outcomes)
            })
        
        return updated
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about learning process"""
        if not self.outcomes:
            return {'status': 'no_outcomes_recorded'}
        
        df = pd.DataFrame(self.outcomes)
        
        return {
            'total_outcomes': len(df),
            'accuracy': df['correct'].mean(),
            'outcomes_by_action': df['predicted_action'].value_counts().to_dict(),
            'outcomes_by_actual': df['actual_outcome'].value_counts().to_dict(),
            'model_updates': len(self.model_updates),
            'recent_trend': df['correct'].tail(100).mean() if len(df) >= 100 else None
        }