# layer2/core/ensemble.py
"""
Ensemble: Combine all stochastic models
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .markov_chain import MarkovChainEngine
from .hmm import FraudHMM
from .survival import SurvivalAnalyzer
from .gambler_ruin import GamblerRuin
from .mdp import InvestigationMDP

logger = logging.getLogger("layer2.ensemble")

class StochasticEnsemble:
    """
    Coordinates all Layer 2 models and produces final predictions
    """
    
    def __init__(self):
        self.markov = MarkovChainEngine()
        self.hmm = FraudHMM()
        self.survival = SurvivalAnalyzer()
        self.gambler = GamblerRuin()
        self.mdp = InvestigationMDP()
        
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, 
            hmm_features: List[str],
            survival_config: Dict = None):
        """
        Fit all models on training data
        
        Args:
            df: Processed DataFrame from Layer 1
            hmm_features: Feature columns for HMM
            survival_config: Optional config for survival model
        """
        logger.info("Fitting Stochastic Ensemble...")
        
        # 1. Fit Markov Chain
        self.markov.fit(df)
        
        # 2. Fit HMM
        self.hmm.fit(df, hmm_features)
        
        # 3. Fit Survival (if time data available)
        if survival_config and 'duration_col' in survival_config:
            self.survival.fit(df, **survival_config)
        
        self.is_fitted = True
        logger.info("Ensemble fitted successfully")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all claims
        
        Returns DataFrame with all model outputs
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        results = df.copy()
        
        # 1. HMM: Fraud probability
        results['fraud_probability'] = self.hmm.get_fraud_probability(df)
        results['hmm_state'] = self.hmm.get_state_labels(df)
        
        # 2. Markov: Absorption probabilities
        absorption_probs = []
        for _, row in results.iterrows():
            state = row['markov_state']
            probs = self.markov.absorption_probabilities(state)
            absorption_probs.append(probs)
        
        abs_df = pd.DataFrame(absorption_probs)
        results = pd.concat([results, abs_df.add_prefix('absorption_prob_')], axis=1)
        
        # 3. Markov: Expected absorption time
        results['expected_absorption_time'] = results['markov_state'].apply(
            lambda s: self.markov.expected_absorption_time(s)
        )
        
        # 4. Gambler's Ruin: Investigation depth
        ruin_results = []
        for _, row in results.iterrows():
            initial_evidence = int(row['red_flag_count'])
            self.gambler.set_evidence_probability(row['fraud_probability'])
            ruin_res = self.gambler.calculate_ruin_probability(initial_evidence)
            ruin_results.append({
                'P_fraud_detected_gr': ruin_res['P_fraud_detected'],
                'P_cleared_gr': ruin_res['P_cleared'],
                'expected_investigation_steps': ruin_res['expected_investigation_steps'],
                'investigation_recommendation': ruin_res['optimal_stopping_point']
            })
        
        ruin_df = pd.DataFrame(ruin_results)
        results = pd.concat([results, ruin_df], axis=1)
        
        # 5. MDP: Optimal action
        self.mdp.value_iteration(
            fraud_prob=results['fraud_probability'].mean(),
            claim_amount=results['total_claim_amount'].mean()
        )
        
        results['optimal_action'] = results['fraud_probability'].apply(
            lambda p: self.mdp.get_action_for_claim(p)
        )
        
        # 6. Survival: Expected resolution time (if fitted)
        if self.survival.is_fitted:
            results['expected_resolution_time'] = self.survival.predict_expectation(df)
        
        logger.info(f"Predictions generated for {len(results)} claims")
        
        return results
    
    def explain(self, claim_row: pd.Series) -> Dict:
        """
        Explain prediction for single claim
        """
        explanation = {
            'claim_id': claim_row.get('claim_id', 'unknown'),
            'markov_state': claim_row['markov_state'],
            'fraud_probability': float(claim_row['fraud_probability']),
            'hmm_latent_state': claim_row['hmm_state'],
            'absorption_probabilities': {
                'Approved': float(claim_row.get('absorption_prob_Approved', 0)),
                'Denied': float(claim_row.get('absorption_prob_Denied', 0)),
                'Fraud_Detected': float(claim_row.get('absorption_prob_Fraud_Detected', 0))
            },
            'expected_absorption_time': float(claim_row['expected_absorption_time']),
            'gambler_ruin_analysis': {
                'P_fraud_detected': float(claim_row['P_fraud_detected_gr']),
                'investigation_steps': float(claim_row['expected_investigation_steps']),
                'recommendation': claim_row['investigation_recommendation']
            },
            'optimal_action': claim_row['optimal_action'],
            'key_features': {
                'red_flag_count': int(claim_row['red_flag_count']),
                'claim_to_premium_ratio': float(claim_row['claim_to_premium_ratio']),
                'severity_score': int(claim_row['severity_score'])
            }
        }
        
        return explanation
    
    def save_models(self, path: str):
        """Save all models to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.hmm.save(f"{path}/hmm.pkl")
        # Add other model saves as needed
        logger.info(f"Models saved to {path}")
    
    def summary(self) -> Dict:
        """Summary of all models"""
        return {
            'markov_chain': self.markov.summary(),
            'hmm': self.hmm.summary(),
            'survival': self.survival.summary() if self.survival.is_fitted else {'status': 'Not fitted'},
            'fitted': self.is_fitted
        }