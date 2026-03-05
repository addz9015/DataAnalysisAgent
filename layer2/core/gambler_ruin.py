# layer2/core/gambler_ruin.py
"""
Gambler's Ruin model for optimal investigation depth
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger("layer2.gambler_ruin")

class GamblerRuin:
    """
    Model investigation as random walk with absorbing barriers
    """
    
    def __init__(self):
        self.p = 0.6  # Probability of evidence pointing to fraud
        self.q = 0.4  # Probability of evidence pointing to legitimate
        
    def calculate_ruin_probability(self, initial_evidence: int, 
                                   fraud_threshold: int = 5,
                                   clearance_threshold: int = -5) -> Dict[str, float]:
        """
        Calculate probability of reaching fraud detection vs clearance
        
        Args:
            initial_evidence: Starting position (0 = neutral, positive = suspicious)
            fraud_threshold: Upper absorbing barrier (+5 = confirmed fraud)
            clearance_threshold: Lower absorbing barrier (-5 = cleared)
        
        Returns:
            Dict with probabilities and expected duration
        """
        i = initial_evidence
        N = fraud_threshold
        M = abs(clearance_threshold)  # 5
        
        if self.p == self.q:  # p = 0.5
            P_fraud = (i + M) / (N + M)
        else:
            # Formula: (1 - (q/p)^i) / (1 - (q/p)^N) when starting from 0
            # Adjust for asymmetric barriers
            r = self.q / self.p
            
            if r == 1:
                P_fraud = (i + M) / (N + M)
            else:
                P_fraud = (1 - r**(i + M)) / (1 - r**(N + M))
        
        P_clearance = 1 - P_fraud
        
        # Expected duration (number of investigation steps)
        if self.p == self.q:
            E_steps = (i + M) * (N - i)
        else:
            # Approximate formula for expected time
            E_steps = ((i + M) / (1 - 2*self.p)) - ((N + M) / (1 - 2*self.p)) * P_fraud
        
        return {
            "P_fraud_detected": float(P_fraud),
            "P_cleared": float(P_clearance),
            "expected_investigation_steps": float(abs(E_steps)),
            "optimal_stopping_point": self._optimal_stopping(i, N, M)
        }
    
    def _optimal_stopping(self, current: int, N: int, M: int) -> str:
        """Recommend whether to continue or stop investigation"""
        # Calculate probabilities once without calling this method again
        res = self._calculate_basic_probs(current, N, M)
        P_fraud = res["P_fraud_detected"]
        
        if P_fraud > 0.8:
            return "Recommend: Intensive investigation (high fraud probability)"
        elif P_fraud > 0.5:
            return "Recommend: Continue standard investigation"
        elif P_fraud > 0.2:
            return "Recommend: Light investigation or fast-track"
        else:
            return "Recommend: Approve claim (low risk)"
            
    def _calculate_basic_probs(self, i: int, N: int, M: int) -> Dict[str, float]:
        """Internal helper for probabilities to avoid recursion"""
        if self.p == self.q:
            P_fraud = (i + M) / (N + M)
        else:
            r = self.q / self.p
            if r == 1:
                P_fraud = (i + M) / (N + M)
            else:
                P_fraud = (1 - r**(i + M)) / (1 - r**(N + M))
        
        return {"P_fraud_detected": float(P_fraud)}
    
    def set_evidence_probability(self, p_fraud: float):
        """Update probability of fraud evidence based on claim features"""
        self.p = np.clip(p_fraud, 0.1, 0.9)
        self.q = 1 - self.p
    
    def sensitivity_analysis(self, initial_evidence: int) -> pd.DataFrame:
        """Analyze how P(fraud) changes with different p values"""
        import pandas as pd
        
        results = []
        original_p = self.p
        
        for p in np.linspace(0.1, 0.9, 9):
            self.p = p
            self.q = 1 - p
            result = self.calculate_ruin_probability(initial_evidence)
            results.append({
                'p_fraud_evidence': p,
                'P_fraud_detected': result['P_fraud_detected'],
                'P_cleared': result['P_cleared']
            })
        
        self.p = original_p
        self.q = 1 - original_p
        
        return pd.DataFrame(results)