# layer2/core/hmm.py
"""
Hidden Markov Model for fraud pattern detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import List, Dict, Optional, Tuple
import logging
import pickle

logger = logging.getLogger("layer2.hmm")

class FraudHMM:
    """
    HMM with 3 latent states: Legitimate, Suspicious, Fraudulent
    """
    
    LATENT_STATES = {
        0: 'Legitimate',
        1: 'Suspicious', 
        2: 'Fraudulent'
    }
    
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=random_state,
            init_params='stmc'  # startprob, transmat, means, covars
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Train HMM on claim features
        
        Args:
            df: DataFrame with engineered features
            feature_cols: List of column names to use
        """
        logger.info(f"Training HMM on {len(feature_cols)} features...")
        
        self.feature_columns = feature_cols
        
        # Extract and scale features
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit HMM
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"HMM fitted. Log likelihood: {self.model.monitor_.history[-1]:.2f}")
        
        # Log learned parameters
        logger.debug(f"Start probs: {self.model.startprob_}")
        logger.debug(f"Transmat:\n{self.model.transmat_}")
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Posterior probabilities of latent states
        
        Returns:
            Array of shape (n_samples, 3) with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def decode(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Viterbi decoding: most likely state sequence
        
        Returns:
            (state_sequence, log_likelihood)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        return self.model.decode(X_scaled)
    
    def get_fraud_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Probability of Fraudulent state (state 2)
        
        Returns:
            Array of fraud probabilities (0-1)
        """
        proba = self.predict_proba(df)
        return proba[:, 2]  # Index 2 = Fraudulent
    
    def get_state_labels(self, df: pd.DataFrame) -> List[str]:
        """Get human-readable state labels for each claim"""
        _, states = self.decode(df)
        return [self.LATENT_STATES[s] for s in states]
    
    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples from HMM
        
        Returns:
            (observations, hidden_states)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        return self.model.sample(n_samples, random_state=random_state)
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance using mean differences between states
        
        Returns dict: {feature: importance_score}
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Use means of each component as proxy for importance
        means = self.model.means_  # Shape: (n_components, n_features)
        
        # Calculate variance of means across states for each feature
        importance = np.var(means, axis=0)
        
        return {
            self.feature_columns[i]: float(importance[i])
            for i in range(len(self.feature_columns))
        }
    
    def save(self, filepath: str):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"HMM saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(n_components=data['model'].n_components)
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_columns = data['feature_columns']
        instance.is_fitted = data['is_fitted']
        
        return instance
    
    def summary(self) -> Dict:
        """Model summary"""
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "n_components": self.n_components,
            "n_features": len(self.feature_columns) if self.feature_columns else 0,
            "convergence": self.model.monitor_.converged,
            "n_iter": self.model.n_iter,
            "log_likelihood": float(self.model.monitor_.history[-1]) if self.model.monitor_.history else None,
            "transition_matrix": self.model.transmat_.tolist(),
            "latent_states": self.LATENT_STATES
        }