# layer1/utils/data_quality.py
"""
Additional data quality checks beyond schema validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataQualityChecker:
    """Advanced data quality assessment"""
    
    @staticmethod
    def check_outliers(df: pd.DataFrame, column: str, 
                      method: str = 'iqr') -> pd.Series:
        """Identify outliers using IQR or Z-score"""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            return z_scores > 3
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, 
                        subset: List[str] = None) -> pd.DataFrame:
        """Check for duplicate records"""
        return df[df.duplicated(subset=subset, keep=False)]
    
    @staticmethod
    def assess_completeness(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness score for each column"""
        return {
            col: (1 - df[col].isna().mean()) * 100 
            for col in df.columns
        }