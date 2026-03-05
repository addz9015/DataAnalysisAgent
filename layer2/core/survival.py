# layer2/core/survival.py
"""
Survival analysis for time-to-absorption (resolution time)
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("layer2.survival")

class SurvivalAnalyzer:
    """
    Cox proportional hazards for claim resolution time
    """
    
    def __init__(self):
        self.cox_model = CoxPHFitter()
        self.km_fitter = KaplanMeierFitter()
        self.is_fitted = False
        self.duration_col = None
        self.event_col = None
        
    def fit(self, df: pd.DataFrame, 
            duration_col: str = 'months_to_resolution',
            event_col: str = 'resolved',
            covariates: Optional[List[str]] = None):
        """
        Fit Cox PH model
        
        Args:
            df: DataFrame with time-to-event data
            duration_col: Column with time duration
            event_col: Column indicating if event occurred (1) or censored (0)
            covariates: List of covariate columns (if None, uses all numeric)
        """
        logger.info("Fitting survival model...")
        
        self.duration_col = duration_col
        self.event_col = event_col
        
        # Select covariates
        if covariates is None:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            covariates = [c for c in covariates if c not in [duration_col, event_col]]
        
        # Prepare data for lifelines
        survival_df = df[[duration_col, event_col] + covariates].copy()
        
        # Fit Cox model
        self.cox_model.fit(survival_df, duration_col=duration_col, event_col=event_col)
        self.is_fitted = True
        
        logger.info(f"Survival model fitted with {len(covariates)} covariates")
        logger.debug(f"Concordance: {self.cox_model.concordance_index_:.3f}")
        
        return self
    
    def predict_survival_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict survival function S(t) for each claim
        
        Returns DataFrame with survival probabilities over time
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        return self.cox_model.predict_survival_function(df)
    
    def predict_median_lifetime(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict median time to resolution
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        return self.cox_model.predict_median(df)
    
    def predict_expectation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict expected (mean) time to resolution
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Use numerical integration of survival function
        survival_funcs = self.predict_survival_function(df)
        
        # Approximate E[T] = integral of S(t) dt
        expectations = []
        for col in survival_funcs.columns:
            sf = survival_funcs[col].values
            times = survival_funcs.index.values
            # Trapezoidal integration
            exp_time = np.trapz(sf, times)
            expectations.append(exp_time)
        
        return np.array(expectations)
    
    def hazard_ratio(self, covariate: str) -> float:
        """
        Get hazard ratio for a specific covariate
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        return np.exp(self.cox_model.params_[covariate])
    
    def summary(self) -> Dict:
        """Model summary"""
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "concordance_index": float(self.cox_model.concordance_index_),
            "log_likelihood": float(self.cox_model.log_likelihood_),
            "AIC": float(self.cox_model.AIC_),
            "n_params": int(self.cox_model.params_.shape[0]),
            "hazard_ratios": {
                cov: float(np.exp(self.cox_model.params_[cov]))
                for cov in self.cox_model.params_.index
            }
        }