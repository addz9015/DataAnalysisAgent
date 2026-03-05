# layer1/core/preprocessing.py
"""
Stage 3: Data Preprocessing
Feature engineering, encoding, scaling, and Markov state derivation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from layer1.config.settings import FEATURE_SETTINGS

logger = logging.getLogger("layer1.preprocessing")

class DataPreprocessor:
    """
    Comprehensive preprocessing pipeline for insurance claims
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_stats = {}
        self.processed_df = None
        
        # Settings from config
        self.severity_mapping = FEATURE_SETTINGS['severity_mapping']
        self.tenure_bins = FEATURE_SETTINGS['tenure_bins']
        self.tenure_labels = FEATURE_SETTINGS['tenure_labels']
        
        logger.info("Preprocessor initialized")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline
        
        Steps:
        1. Handle missing values
        2. Engineer features
        3. Encode categoricals
        4. Derive Markov states
        5. Scale numerical features
        """
        logger.info(f"Starting preprocessing of {len(df)} records")
        
        df = df.copy()
        initial_cols = set(df.columns)
        
        # Step 1: Handle missing values
        df = self._handle_missing(df)
        
        # Step 2: Feature engineering
        df = self._engineer_features(df)
        
        # Step 3: Encode categoricals
        df = self._encode_categoricals(df)
        
        # Step 4: Derive Markov states (CRITICAL)
        df = self._derive_markov_states(df)
        
        # Step 5: Scale features
        df = self._scale_features(df)
        
        # Track feature statistics
        new_cols = set(df.columns) - initial_cols
        self.feature_stats = {
            'original_features': len(initial_cols),
            'engineered_features': len(new_cols),
            'total_features': len(df.columns),
            'new_columns': list(new_cols)
        }
        
        self.processed_df = df
        logger.info(f"Preprocessing complete. Features: {self.feature_stats}")
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        df = df.copy()
        
        # Numeric columns: median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Imputed {col} with median: {median_val}")
        
        # Categorical columns: mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                logger.debug(f"Imputed {col} with mode: {mode_val}")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for modeling"""
        df = df.copy()
        
        # Financial ratios
        df['claim_to_premium_ratio'] = df['total_claim_amount'] / df['policy_annual_premium']
        df['claim_to_premium_ratio'] = df['claim_to_premium_ratio'].replace([np.inf, -np.inf], 0)
        
        # Component ratios
        df['injury_ratio'] = (df['injury_claim'] / df['total_claim_amount']).fillna(0)
        df['property_ratio'] = (df['property_claim'] / df['total_claim_amount']).fillna(0)
        df['vehicle_ratio'] = (df['vehicle_claim'] / df['total_claim_amount']).fillna(0)
        
        # Severity score
        df['severity_score'] = df['incident_severity'].map(self.severity_mapping)
        
        # Customer tenure category
        df['customer_tenure'] = pd.cut(
            df['months_as_customer'],
            bins=self.tenure_bins,
            labels=self.tenure_labels,
            include_lowest=True
        )
        
        # Map witnesses count to binary presence if missing
        if 'witness_present' not in df.columns and 'witnesses' in df.columns:
            df['witness_present'] = df['witnesses'].apply(lambda x: 'Yes' if x > 0 else 'No')

        # Red flag indicators (fraud signals)
        df['red_flag_high_claim'] = (df['claim_to_premium_ratio'] > 10).astype(int)
        df['red_flag_no_witness'] = (df['witness_present'].isin(['No', 'N', '0', 'no', 'n'])).astype(int)
        df['red_flag_no_police'] = (df['police_report_available'].isin(['No', 'N', '0'])).astype(int)
        df['red_flag_no_authorities'] = (df['authorities_contacted'] == 'None').astype(int)
        
        # Total red flag count
        df['red_flag_count'] = (
            df['red_flag_high_claim'] +
            df['red_flag_no_witness'] +
            df['red_flag_no_police'] +
            df['red_flag_no_authorities']
        )
        
        # Claim complexity score
        df['complexity_score'] = (
            df['severity_score'] * 0.4 +
            df['red_flag_count'] * 0.3 +
            (df['total_claim_amount'] / 10000).clip(0, 5) * 0.3
        )
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 40, 60, 100],
            labels=['Young', 'Adult', 'Middle', 'Senior']
        )
        
        logger.debug(f"Engineered {len(df.columns) - 15} new features")
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for ML models"""
        df = df.copy()
        
        # Label encoding for ordinal categories
        ordinal_mappings = {
            'incident_severity': self.severity_mapping,
            'customer_tenure': {label: idx for idx, label in enumerate(self.tenure_labels)},
            'age_group': {'Young': 0, 'Adult': 1, 'Middle': 2, 'Senior': 3}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].map(mapping)
        
        # Binary encoding
        binary_mappings = {
            'witness_present': {'Yes': 1, 'Y': 1, '1': 1, 'No': 0, 'N': 0, '0': 0},
            'police_report_available': {
                'Yes': 1, 'Y': 1, '1': 1, 
                'No': 0, 'N': 0, '0': 0, 
                'Unknown': -1, 'U': -1
            }
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(-1)
        
        # One-hot encoding for nominal categories
        nominal_cols = ['incident_type', 'collision_type', 'authorities_contacted']
        for col in nominal_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _derive_markov_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL: Derive Markov chain states from claim features
        This connects raw data to stochastic models in Layer 2
        """
        df = df.copy()
        
        def assign_state(row):
            """Assign Markov state based on claim characteristics"""
            
            # Check for absorbing states first (terminal states)
            if row.get('fraud_reported') in ['Y', 'Yes', '1']:
                return 'Fraud_Detected'
            
            if row.get('claim_status') == 'Closed':
                settlement = row.get('settlement_amount', 0) or 0
                return 'Approved' if settlement > 0 else 'Denied'
            
            # 2. Check Transient States (v1.1 Rules)
            red_flags = row.get('red_flag_count', 0)
            severity = row.get('severity_score', 0)
            
            # Fast_Track: red_flag_count = 0 AND severity_score <= 2
            if red_flags == 0 and severity <= 2:
                return 'Fast_Track'
            
            # High_Risk_Investigation: red_flag_count >= 3
            if red_flags >= 3:
                return 'High_Risk_Investigation'
            
            # Complex_Review: severity_score >= 3 AND red_flag_count <= 1
            if severity >= 3 and red_flags <= 1:
                return 'Complex_Review'
            
            # Standard_Investigation: red_flag_count = 1 OR (red_flag_count = 2 AND severity_score <= 2)
            if red_flags == 1 or (red_flags == 2 and severity <= 2):
                return 'Standard_Investigation'
            
            # Default to Standard if no rules match perfectly (fallback)
            return 'Standard_Investigation'
        
        # Apply state assignment
        df['markov_state'] = df.apply(assign_state, axis=1)
        
        # Create numerical encoding for algorithms
        state_order = [
            'Fast_Track', 'Complex_Review', 
            'Standard_Investigation', 'High_Risk_Investigation',
            'Approved', 'Denied', 'Fraud_Detected'
        ]
        df['markov_state_encoded'] = df['markov_state'].map(
            {state: idx for idx, state in enumerate(state_order)}
        )
        
        # Mark state type (transient vs absorbing)
        absorbing_states = ['Approved', 'Denied', 'Fraud_Detected']
        df['is_absorbing'] = df['markov_state'].isin(absorbing_states).astype(int)
        
        logger.info(f"Markov states distribution:\n{df['markov_state'].value_counts()}")
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features for ML algorithms"""
        df = df.copy()
        
        # Select numerical features to scale
        scale_cols = [
            'age', 'policy_annual_premium', 'total_claim_amount',
            'claim_to_premium_ratio', 'injury_ratio', 'property_ratio',
            'vehicle_ratio', 'severity_score', 'complexity_score'
        ]
        
        # Only scale columns that exist
        scale_cols = [col for col in scale_cols if col in df.columns]
        
        if scale_cols:
            # Fit and transform
            scaled_values = self.scaler.fit_transform(df[scale_cols])
            
            # Replace with scaled values
            for idx, col in enumerate(scale_cols):
                df[f'{col}_scaled'] = scaled_values[:, idx]
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Organize features by model type for Layer 2"""
        return {
            'markov_chain': [
                'markov_state', 'markov_state_encoded', 'is_absorbing',
                'months_as_customer', 'customer_tenure_encoded'
            ],
            'hmm': sorted([col for col in self.processed_df.columns if col.endswith('_scaled') or col.endswith('_encoded') or any(p in col for p in ['incident_type_', 'collision_type_', 'authorities_contacted_']) or col == 'red_flag_count']),
            'mdp': [
                'total_claim_amount_scaled', 'policy_annual_premium_scaled',
                'red_flag_count', 'complexity_score_scaled'
            ],
            'metadata': [
                'claim_id', '_ingestion_timestamp', '_batch_id'
            ]
        }