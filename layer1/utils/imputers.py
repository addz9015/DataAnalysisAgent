# layer1/utils/imputers.py
"""
Advanced imputation strategies for missing values
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class AdvancedImputers:
    """
    Advanced missing value imputation strategies
    """
    
    @staticmethod
    def impute_by_group(df: pd.DataFrame,
                       column: str,
                       group_by: str,
                       strategy: str = 'median') -> pd.Series:
        """
        Impute missing values using group statistics
        
        Example: Impute missing premiums by customer tenure group
        """
        def impute_group(group):
            if strategy == 'median':
                fill_val = group.median()
            elif strategy == 'mean':
                fill_val = group.mean()
            elif strategy == 'mode':
                fill_val = group.mode()[0] if not group.mode().empty else group.dropna().iloc[0]
            else:
                fill_val = group.median()
            
            return group.fillna(fill_val)
        
        return df.groupby(group_by)[column].transform(impute_group)
    
    @staticmethod
    def impute_with_indicator(df: pd.DataFrame,
                             column: str,
                             strategy: str = 'median') -> pd.DataFrame:
        """
        Impute missing values and add missing indicator column
        
        Useful when missingness itself is informative (e.g., missing police report)
        """
        df = df.copy()
        
        # Add missing indicator
        indicator_col = f'{column}_missing'
        df[indicator_col] = df[column].isna().astype(int)
        
        # Impute
        if strategy == 'median':
            fill_val = df[column].median()
        elif strategy == 'mean':
            fill_val = df[column].mean()
        elif strategy == 'zero':
            fill_val = 0
        else:
            fill_val = df[column].median()
        
        df[column].fillna(fill_val, inplace=True)
        
        return df
    
    @staticmethod
    def knn_impute(df: pd.DataFrame,
                   columns: List[str],
                   n_neighbors: int = 5) -> pd.DataFrame:
        """
        K-Nearest Neighbors imputation for multivariate missing values
        
        Uses similarity between records to impute
        """
        df = df.copy()
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[columns] = imputer.fit_transform(df[columns])
        
        return df
    
    @staticmethod
    def iterative_impute(df: pd.DataFrame,
                        columns: List[str],
                        max_iter: int = 10) -> pd.DataFrame:
        """
        Iterative imputation (MICE - Multiple Imputation by Chained Equations)
        
        Models each feature with missing values as function of other features
        """
        df = df.copy()
        
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        df[columns] = imputer.fit_transform(df[columns])
        
        return df
    
    @staticmethod
    def impute_categorical_by_target(df: pd.DataFrame,
                                    categorical_col: str,
                                    target_col: str,
                                    strategy: str = 'mode') -> pd.Series:
        """
        Impute categorical missing values based on target variable distribution
        
        Example: If fraud cases usually have 'No' witness, impute accordingly
        """
        def impute_by_target(group):
            if strategy == 'mode':
                return group.mode()[0] if not group.mode().empty else 'Unknown'
            elif strategy == 'first':
                return group.dropna().iloc[0] if not group.dropna().empty else 'Unknown'
            else:
                return group.mode()[0] if not group.mode().empty else 'Unknown'
        
        # Group by target and get imputation value for each group
        impute_map = df.groupby(target_col)[categorical_col].apply(impute_by_target)
        
        # Fill missing values based on target
        def fill_missing(row):
            if pd.isna(row[categorical_col]):
                target_val = row[target_col]
                return impute_map.get(target_val, 'Unknown')
            return row[categorical_col]
        
        return df.apply(fill_missing, axis=1)


class MissingValueAnalyzer:
    """
    Analyze patterns in missing values
    """
    
    @staticmethod
    def missing_pattern(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify patterns of missing values across columns
        
        Returns DataFrame showing which columns tend to be missing together
        """
        # Create binary missing indicator matrix
        missing_matrix = df.isnull().astype(int)
        
        # Find unique patterns
        patterns = missing_matrix.value_counts().reset_index()
        patterns.columns = list(df.columns) + ['count']
        
        return patterns.sort_values('count', ascending=False)
    
    @staticmethod
    def missing_correlation(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between missing indicators of different columns
        
        High correlation suggests non-random missingness (MNAR - Missing Not At Random)
        """
        missing_matrix = df.isnull().astype(int)
        return missing_matrix.corr()
    
    @staticmethod
    def recommend_imputation_strategy(df: pd.DataFrame, 
                                     column: str) -> Dict[str, Any]:
        """
        Recommend best imputation strategy based on data characteristics
        """
        missing_pct = df[column].isna().mean() * 100
        dtype = df[column].dtype
        
        recommendation = {
            'column': column,
            'missing_percentage': missing_pct,
            'dtype': str(dtype),
            'recommended_strategy': None,
            'reasoning': None
        }
        
        if missing_pct > 50:
            recommendation['recommended_strategy'] = 'drop_column'
            recommendation['reasoning'] = 'Too many missing values (>50%)'
        
        elif dtype in ['int64', 'float64']:
            skewness = df[column].skew()
            
            if abs(skewness) > 2:
                recommendation['recommended_strategy'] = 'median'
                recommendation['reasoning'] = f'Highly skewed distribution (skew={skewness:.2f})'
            else:
                recommendation['recommended_strategy'] = 'mean'
                recommendation['reasoning'] = 'Approximately normal distribution'
        
        else:
            recommendation['recommended_strategy'] = 'mode'
            recommendation['reasoning'] = 'Categorical variable'
        
        return recommendation