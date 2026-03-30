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
        self.fraud_state_idx = 2
        self.legit_state_idx = 0
        self.state_label_map = dict(self.LATENT_STATES)
        
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

        # Latent-state indices are arbitrary in unsupervised HMMs.
        # Infer semantics from labeled fraud outcomes when available.
        self._infer_state_semantics(df, X_scaled)
        
        logger.info(f"HMM fitted. Log likelihood: {self.model.monitor_.history[-1]:.2f}")
        
        # Log learned parameters
        logger.debug(f"Start probs: {self.model.startprob_}")
        logger.debug(f"Transmat:\n{self.model.transmat_}")
        
        return self

    def _infer_state_semantics(self, df: Optional[pd.DataFrame] = None, X_scaled: Optional[np.ndarray] = None):
        """Infer latent-state semantics; prefer label-anchored mapping with heuristic fallback."""
        if self.feature_columns is None:
            return

        # Preferred path: map states using labeled fraud outcomes from training data.
        if df is not None and X_scaled is not None and 'fraud_reported' in df.columns:
            raw = df['fraud_reported'].astype(str).str.strip().str.upper()
            fraud_mask = raw.isin(['Y', 'YES', '1', 'TRUE'])
            legit_mask = raw.isin(['N', 'NO', '0', 'FALSE'])

            if fraud_mask.any() and legit_mask.any():
                post = self.model.predict_proba(X_scaled)
                fraud_means = post[fraud_mask.values].mean(axis=0)
                legit_means = post[legit_mask.values].mean(axis=0)

                fraud_idx = int(np.argmax(fraud_means))
                legit_idx = int(np.argmax(legit_means))

                # If both classes map to same state, keep fraud state and push legit to the weakest fraud state.
                if legit_idx == fraud_idx:
                    legit_idx = int(np.argmin(fraud_means))

                self.fraud_state_idx = fraud_idx
                self.legit_state_idx = legit_idx

                self.state_label_map = {idx: 'Suspicious' for idx in range(self.n_components)}
                self.state_label_map[self.legit_state_idx] = 'Legitimate'
                self.state_label_map[self.fraud_state_idx] = 'Fraudulent'

                logger.info(
                    "Inferred HMM semantics from labels: legit=%s fraud=%s fraud_means=%s legit_means=%s",
                    self.legit_state_idx,
                    self.fraud_state_idx,
                    np.round(fraud_means, 4).tolist(),
                    np.round(legit_means, 4).tolist(),
                )
                return

        feature_idx = {name: i for i, name in enumerate(self.feature_columns)}
        risk_weights = {
            'red_flag_count': 2.5,
            'claim_to_premium_ratio_scaled': 1.6,
            'total_claim_amount_scaled': 1.0,
            'severity_score_scaled': 1.0,
            'complexity_score_scaled': 1.2,
        }

        state_scores = {}
        for state_idx in range(self.n_components):
            means = self.model.means_[state_idx]
            score = 0.0
            for feat, weight in risk_weights.items():
                idx = feature_idx.get(feat)
                if idx is not None:
                    score += float(means[idx]) * weight
            state_scores[state_idx] = score

        ordered_states = sorted(state_scores, key=state_scores.get)
        self.legit_state_idx = ordered_states[0]
        self.fraud_state_idx = ordered_states[-1]

        # Build human-readable labels by inferred risk ordering.
        self.state_label_map = {idx: 'Suspicious' for idx in range(self.n_components)}
        self.state_label_map[self.legit_state_idx] = 'Legitimate'
        self.state_label_map[self.fraud_state_idx] = 'Fraudulent'

        logger.info(
            "Inferred HMM semantics from heuristic fallback: legit=%s fraud=%s scores=%s",
            self.legit_state_idx,
            self.fraud_state_idx,
            {k: round(v, 4) for k, v in state_scores.items()},
        )
    
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
            Array of fraud probabilities (0-1), calibrated to prevent overfitting
        """
        proba = self.predict_proba(df)
        fraud_probs = proba[:, self.fraud_state_idx]
        
        # Calibration: Compress extreme probabilities to prevent 100% fraud predictions
        # Use Platt-like scaling: shrink toward 0.5 for extreme values
        calibrated = 0.5 + (fraud_probs - 0.5) * 0.8  # Reduce extreme confidence by 20%
        calibrated = np.clip(calibrated, 0.01, 0.99)
        
        return calibrated
    
    def get_state_labels(self, df: pd.DataFrame) -> List[str]:
        """Get human-readable state labels for each claim"""
        _, states = self.decode(df)
        return [self.state_label_map.get(s, self.LATENT_STATES.get(s, 'Unknown')) for s in states]
    
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
                'is_fitted': self.is_fitted,
                'fraud_state_idx': self.fraud_state_idx,
                'legit_state_idx': self.legit_state_idx,
                'state_label_map': self.state_label_map,
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
        instance.fraud_state_idx = data.get('fraud_state_idx', 2)
        instance.legit_state_idx = data.get('legit_state_idx', 0)
        instance.state_label_map = data.get('state_label_map', dict(cls.LATENT_STATES))
        
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