# layer2/core/markov_chain.py
"""
Markov Chain Engine for claim state transitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from ..utils.matrix_ops import check_ergodicity, compute_stationary_distribution

logger = logging.getLogger("layer2.markov")

class MarkovChainEngine:
    """
    Analyzes claim transitions and calculates absorption probabilities
    """
    
    def __init__(self):
        self.states = []
        self.transition_matrix = None
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.is_fitted = False
        
        # Define known absorbing states
        self.absorbing_threshold = 0.95
        
    def fit(self, df: pd.DataFrame, state_col: str = 'markov_state'):
        """
        Estimate transition matrix from state data
        In this specific implementation, we use the distribution of states 
        and domain knowledge to build the transition probabilities
        """
        logger.info("Fitting Markov Chain Engine...")
        
        if state_col not in df.columns:
            raise ValueError(f"Column {state_col} not found in DataFrame")
            
        self.states = sorted(df[state_col].unique().tolist())
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for state, i in self.state_to_idx.items()}
        n = len(self.states)
        
        # Initialize transition matrix
        # For a single-snapshot dataset, we estimate transitions based on state logic
        P = np.zeros((n, n))
        
        # State distribution for prior-based estimation
        dist = df[state_col].value_counts(normalize=True).to_dict()
        
        for i, state in enumerate(self.states):
            if state == 'Fraud_Detected':
                P[i, i] = 1.0  # Absorbing
            elif state == 'Denied':
                P[i, i] = 1.0  # Absorbing
            elif state == 'Approved':
                P[i, i] = 1.0  # Absorbing
            elif state == 'Fast_Track':
                # Fast track usually leads to Approval or Investigation
                P[i, self.state_to_idx.get('Approved', i)] = 0.7
                P[i, self.state_to_idx.get('Standard_Investigation', i)] = 0.2
                P[i, i] = 0.1
            elif state == 'Standard_Investigation':
                P[i, self.state_to_idx.get('Approved', i)] = 0.4
                P[i, self.state_to_idx.get('Complex_Review', i)] = 0.3
                P[i, self.state_to_idx.get('Denied', i)] = 0.2
                P[i, i] = 0.1
            elif state == 'Complex_Review':
                P[i, self.state_to_idx.get('Fraud_Detected', i)] = 0.4
                P[i, self.state_to_idx.get('Denied', i)] = 0.3
                P[i, self.state_to_idx.get('Approved', i)] = 0.2
                P[i, i] = 0.1
            else:
                # Default behavior
                P[i, i] = 1.0
                
        # Normalize rows to ensure stochasticity
        row_sums = P.sum(axis=1)
        self.transition_matrix = P / row_sums[:, np.newaxis]
        
        self.is_fitted = True
        logger.info(f"Markov Chain fitted with {n} states")
        
        return self
        
    def absorption_probabilities(self, start_state: str) -> Dict[str, float]:
        """
        Calculate probability of ending in each absorbing state
        Using Fundamental Matrix: B = (I - Q)^-1 * R
        """
        if not self.is_fitted:
            raise RuntimeError("Engine not fitted")
            
        if start_state not in self.state_to_idx:
            return {s: 0.0 for s in self.states if self._is_absorbing_by_name(s)}
            
        # Identify absorbing and transient states
        absorbing_mask = np.array([self._is_absorbing_by_idx(i) for i in range(len(self.states))])
        transient_mask = ~absorbing_mask
        
        if absorbing_mask[self.state_to_idx[start_state]]:
            # Already in an absorbing state
            res = {self.idx_to_state[i]: 0.0 for i in np.where(absorbing_mask)[0]}
            res[start_state] = 1.0
            return res
            
        # Extract Q (transient to transient) and R (transient to absorbing)
        Q = self.transition_matrix[np.ix_(transient_mask, transient_mask)]
        R = self.transition_matrix[np.ix_(transient_mask, absorbing_mask)]
        
        # Fundamental Matrix N = (I - Q)^-1
        I = np.eye(Q.shape[0])
        try:
            N = np.linalg.inv(I - Q)
            # B = N * R
            B = np.matmul(N, R)
            
            # Get probabilities for the specific start state
            transient_idx = np.where(np.where(transient_mask)[0] == self.state_to_idx[start_state])[0][0]
            probs = B[transient_idx, :]
            
            absorbing_indices = np.where(absorbing_mask)[0]
            return {self.idx_to_state[idx]: float(probs[i]) for i, idx in enumerate(absorbing_indices)}
        except np.linalg.LinAlgError:
            logger.error("Matrix (I-Q) is singular, cannot calculate absorption")
            return {}

    def expected_absorption_time(self, start_state: str) -> float:
        """
        Calculate expected number of steps until absorption
        E = N * 1 (column vector of ones)
        """
        if not self.is_fitted:
            raise RuntimeError("Engine not fitted")
            
        if start_state not in self.state_to_idx:
            return 0.0
            
        absorbing_mask = np.array([self._is_absorbing_by_idx(i) for i in range(len(self.states))])
        transient_mask = ~absorbing_mask
        
        if absorbing_mask[self.state_to_idx[start_state]]:
            return 0.0
            
        Q = self.transition_matrix[np.ix_(transient_mask, transient_mask)]
        I = np.eye(Q.shape[0])
        
        try:
            N = np.linalg.inv(I - Q)
            steps = np.sum(N, axis=1)
            
            transient_idx = np.where(np.where(transient_mask)[0] == self.state_to_idx[start_state])[0][0]
            return float(steps[transient_idx])
        except np.linalg.LinAlgError:
            return 0.0

    def _is_absorbing_by_idx(self, idx: int) -> bool:
        return self.transition_matrix[idx, idx] >= self.absorbing_threshold

    def _is_absorbing_by_name(self, name: str) -> bool:
        return name in ['Approved', 'Denied', 'Fraud_Detected']

    def summary(self) -> Dict:
        """Model summary"""
        if not self.is_fitted:
            return {"status": "Not fitted"}
            
        return {
            "n_states": len(self.states),
            "states": self.states,
            "transition_matrix": self.transition_matrix.tolist()
        }