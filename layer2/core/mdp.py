# layer2/core/mdp.py
"""
Markov Decision Process for optimal investigation policy
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger("layer2.mdp")

class InvestigationMDP:
    """
    MDP for optimizing investigation actions
    States: Suspicion levels (0-10)
    Actions: Fast-track, Standard, Deep investigation, Approve, Deny
    """
    
    ACTIONS = ['fast_track', 'standard', 'deep', 'approve', 'deny']
    
    def __init__(self, n_states: int = 11):  # 0-10 suspicion levels
        self.n_states = n_states
        self.n_actions = len(self.ACTIONS)
        self.policy = None
        self.value_function = None
        self.is_solved = False
        
        # Cost structure
        self.costs = {
            'fast_track': 50,
            'standard': 200,
            'deep': 1000,
            'approve': 0,
            'deny': 100
        }
        
    def define_rewards(self, fraud_prob: float, claim_amount: float) -> np.ndarray:
        """
        Define reward matrix R[s, a]
        
        Args:
            fraud_prob: Probability claim is fraudulent
            claim_amount: Total claim amount (for loss calculation)
        """
        R = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a_idx, action in enumerate(self.ACTIONS):
                base_cost = -self.costs[action]
                
                if action == 'approve':
                    # If fraudulent: lose claim amount
                    expected_loss = -fraud_prob * claim_amount * 0.8  # 80% recovery if fraud
                    R[s, a_idx] = base_cost + expected_loss
                    
                elif action == 'deny':
                    # If legitimate: lose customer (opportunity cost)
                    expected_loss = -(1 - fraud_prob) * 500
                    R[s, a_idx] = base_cost + expected_loss
                    
                elif action == 'deep':
                    # Deep investigation reveals truth with high probability
                    info_value = fraud_prob * 0.3 * claim_amount
                    R[s, a_idx] = base_cost + info_value
                    
                elif action == 'standard':
                    info_value = fraud_prob * 0.15 * claim_amount
                    R[s, a_idx] = base_cost + info_value
                    
                else:  # fast_track
                    info_value = fraud_prob * 0.05 * claim_amount
                    R[s, a_idx] = base_cost + info_value
        
        return R
    
    def define_transitions(self) -> np.ndarray:
        """
        Define transition probabilities P[s, a, s']
        Simplified: actions move suspicion level or absorb
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s in range(self.n_states):
            for a_idx, action in enumerate(self.ACTIONS):
                if action in ['approve', 'deny']:
                    # Absorbing actions (terminal)
                    P[s, a_idx, s] = 1.0
                    
                elif action == 'fast_track':
                    # Small chance of increasing suspicion
                    if s < self.n_states - 1:
                        P[s, a_idx, s] = 0.9
                        P[s, a_idx, s + 1] = 0.1
                    else:
                        P[s, a_idx, s] = 1.0
                        
                elif action == 'standard':
                    # Moderate information gain
                    if s < self.n_states - 2:
                        P[s, a_idx, s + 1] = 0.3
                        P[s, a_idx, s + 2] = 0.2
                        P[s, a_idx, s] = 0.5
                    else:
                        P[s, a_idx, s] = 1.0
                        
                else:  # deep
                    # High information gain
                    if s < self.n_states - 3:
                        P[s, a_idx, s + 2] = 0.4
                        P[s, a_idx, s + 3] = 0.3
                        P[s, a_idx, s + 1] = 0.2
                        P[s, a_idx, s] = 0.1
                    else:
                        P[s, a_idx, self.n_states - 1] = 1.0
        
        return P
    
    def value_iteration(self, fraud_prob: float, claim_amount: float,
                       gamma: float = 0.95, theta: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve MDP using value iteration
        
        Returns:
            (policy, value_function)
        """
        logger.info("Running value iteration...")
        
        R = self.define_rewards(fraud_prob, claim_amount)
        P = self.define_transitions()
        
        V = np.zeros(self.n_states)
        
        for iteration in range(max_iter):
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                action_values = []
                for a in range(self.n_actions):
                    # Q(s,a) = R(s,a) + gamma * sum(P(s'|s,a) * V(s'))
                    expected_future = np.sum(P[s, a, :] * V)
                    q_value = R[s, a] + gamma * expected_future
                    action_values.append(q_value)
                
                V_new[s] = max(action_values)
            
            # Check convergence
            if np.max(np.abs(V_new - V)) < theta:
                logger.info(f"Value iteration converged in {iteration} iterations")
                break
            
            V = V_new
        
        # Extract policy
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            action_values = []
            for a in range(self.n_actions):
                expected_future = np.sum(P[s, a, :] * V)
                q_value = R[s, a] + gamma * expected_future
                action_values.append(q_value)
            policy[s] = np.argmax(action_values)
        
        self.policy = policy
        self.value_function = V
        self.is_solved = True
        
        return policy, V
    
    def get_optimal_action(self, suspicion_level: int) -> str:
        """Get optimal action for given suspicion level"""
        if not self.is_solved:
            raise RuntimeError("MDP not solved. Run value_iteration() first.")
        
        if suspicion_level < 0 or suspicion_level >= self.n_states:
            raise ValueError(f"Suspicion level must be in [0, {self.n_states-1}]")
        
        action_idx = self.policy[suspicion_level]
        return self.ACTIONS[action_idx]
    
    def get_action_for_claim(self, fraud_probability: float, max_suspicion: int = 10) -> str:
        """Map fraud probability to suspicion level and return action"""
        suspicion_level = int(fraud_probability * max_suspicion)
        suspicion_level = min(suspicion_level, max_suspicion)
        return self.get_optimal_action(suspicion_level)