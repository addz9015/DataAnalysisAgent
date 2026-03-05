# layer1/utils/encoders.py
"""
Custom encoders for categorical and special data types
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import LabelEncoder

class CustomEncoders:
    """
    Custom encoding utilities for insurance claim data
    """
    
    @staticmethod
    def encode_binary_series(series: pd.Series, 
                            true_values: List[str] = None,
                            false_values: List[str] = None,
                            unknown_value: int = -1) -> pd.Series:
        """
        Encode binary categorical series to numeric
        
        Args:
            series: Input categorical series
            true_values: List of values to encode as 1
            false_values: List of values to encode as 0
            unknown_value: Value for unknown categories
        
        Returns:
            Encoded numeric series
        """
        if true_values is None:
            true_values = ['Yes', 'Y', '1', 'True', 'true']
        if false_values is None:
            false_values = ['No', 'N', '0', 'False', 'false']
        
        def encode_value(val):
            if pd.isna(val):
                return unknown_value
            val_str = str(val).strip()
            if val_str in true_values:
                return 1
            elif val_str in false_values:
                return 0
            else:
                return unknown_value
        
        return series.apply(encode_value)
    
    @staticmethod
    def encode_ordinal(series: pd.Series, 
                       mapping: Dict[str, int],
                       default: int = -1) -> pd.Series:
        """
        Encode ordinal categorical series using explicit mapping
        
        Args:
            series: Input categorical series
            mapping: Dictionary mapping categories to numeric values
            default: Default value for unmapped categories
        
        Returns:
            Encoded numeric series
        """
        return series.map(mapping).fillna(default)
    
    @staticmethod
    def frequency_encode(series: pd.Series) -> pd.Series:
        """
        Frequency encoding: replace category with its occurrence count
        
        Useful for high-cardinality categorical features
        """
        freq_map = series.value_counts().to_dict()
        return series.map(freq_map)
    
    @staticmethod
    def target_encode(series: pd.Series, 
                     target: pd.Series,
                     smoothing: float = 1.0) -> pd.Series:
        """
        Target encoding: replace category with mean target value
        
        Args:
            series: Categorical feature
            target: Binary target variable
            smoothing: Smoothing parameter to prevent overfitting
        """
        global_mean = target.mean()
        
        # Calculate mean target per category
        agg = pd.DataFrame({'category': series, 'target': target}).groupby('category')['target'].agg(['mean', 'count'])
        
        # Apply smoothing
        smoothed_mean = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        
        return series.map(smoothed_mean).fillna(global_mean)


class TimeFeatureEncoder:
    """
    Encode temporal features from dates and timestamps
    """
    
    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, 
                                  datetime_col: str,
                                  prefix: Optional[str] = None) -> pd.DataFrame:
        """
        Extract multiple features from datetime column
        
        Returns DataFrame with new time-based features
        """
        df = df.copy()
        prefix = prefix or datetime_col
        
        # Convert to datetime if not already
        dt_series = pd.to_datetime(df[datetime_col], errors='coerce')
        
        # Extract features
        df[f'{prefix}_year'] = dt_series.dt.year
        df[f'{prefix}_month'] = dt_series.dt.month
        df[f'{prefix}_day'] = dt_series.dt.day
        df[f'{prefix}_dayofweek'] = dt_series.dt.dayofweek
        df[f'{prefix}_quarter'] = dt_series.dt.quarter
        df[f'{prefix}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
        df[f'{prefix}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
        df[f'{prefix}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
        
        # Cyclical encoding for month (captures seasonality)
        df[f'{prefix}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
        df[f'{prefix}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
        
        return df
    
    @staticmethod
    def calculate_time_since(df: pd.DataFrame,
                            start_col: str,
                            end_col: str,
                            output_col: str,
                            unit: str = 'days') -> pd.DataFrame:
        """
        Calculate time difference between two datetime columns
        """
        df = df.copy()
        
        start_dt = pd.to_datetime(df[start_col], errors='coerce')
        end_dt = pd.to_datetime(df[end_col], errors='coerce')
        
        diff = end_dt - start_dt
        
        if unit == 'days':
            df[output_col] = diff.dt.total_seconds() / (24 * 3600)
        elif unit == 'hours':
            df[output_col] = diff.dt.total_seconds() / 3600
        elif unit == 'minutes':
            df[output_col] = diff.dt.total_seconds() / 60
        
        return df