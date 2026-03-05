# layer1/tests/test_preprocessing.py
"""
Tests for preprocessing module
"""

import pytest
import pandas as pd
import numpy as np

from layer1.core.preprocessing import DataPreprocessor

class TestDataPreprocessor:
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for preprocessing tests"""
        return pd.DataFrame({
            'months_as_customer': [24, 36, 12, 48, 120],
            'age': [35, 45, 28, 55, 67],
            'policy_annual_premium': [1200.0, 1500.0, 1000.0, 2000.0, 2500.0],
            'incident_severity': ['Minor Damage', 'Major Damage', 'Trivial Damage', 'Total Loss', 'Minor Damage'],
            'total_claim_amount': [5000.0, 15000.0, 800.0, 25000.0, 3000.0],
            'injury_claim': [2000.0, 5000.0, 0.0, 10000.0, 1000.0],
            'property_claim': [2000.0, 5000.0, 500.0, 10000.0, 1500.0],
            'vehicle_claim': [1000.0, 5000.0, 300.0, 5000.0, 500.0],
            'incident_type': ['Single Vehicle', 'Multi-Vehicle', 'Parked Car', 'Vehicle Theft', 'Single Vehicle'],
            'collision_type': ['Front Collision', 'Rear Collision', 'No Collision', 'Side Collision', 'Front Collision'],
            'authorities_contacted': ['Police', 'Police', 'None', 'Fire', 'Police'],
            'witness_present': ['Yes', 'No', 'No', 'Yes', 'No'],
            'police_report_available': ['Yes', 'Yes', 'No', 'Unknown', 'Yes'],
            'fraud_reported': ['No', 'No', 'No', 'Yes', 'No'],
            'claim_id': ['C001', 'C002', 'C003', 'C004', 'C005']
        })
    
    def test_feature_engineering(self, sample_df):
        """Test that new features are created"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        # Check engineered features exist
        assert 'claim_to_premium_ratio' in result.columns
        assert 'injury_ratio' in result.columns
        assert 'red_flag_count' in result.columns
        assert 'severity_score' in result.columns
    
    def test_claim_to_premium_ratio(self, sample_df):
        """Test claim to premium ratio calculation"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        # First row: 5000 / 1200 = 4.17
        expected_ratio = 5000.0 / 1200.0
        assert abs(result.loc[0, 'claim_to_premium_ratio'] - expected_ratio) < 0.01
    
    def test_red_flag_count(self, sample_df):
        """Test red flag counting"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        # Row 2: No witness, No police, No authorities = 3 red flags
        # Plus high claim ratio (800/1000 = 0.8, not > 10) = 0
        # Total = 3
        assert result.loc[2, 'red_flag_count'] == 3
    
    def test_markov_state_derivation(self, sample_df):
        """Test Markov state assignment"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        assert 'markov_state' in result.columns
        assert 'markov_state_encoded' in result.columns
        assert 'is_absorbing' in result.columns
        
        # Row 3 has fraud_reported = 'Yes', should be Fraud_Detected
        assert result.loc[3, 'markov_state'] == 'Fraud_Detected'
        assert result.loc[3, 'is_absorbing'] == 1
    
    def test_encoding(self, sample_df):
        """Test categorical encoding"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        # Check encoded columns exist
        assert 'incident_severity_encoded' in result.columns
        assert 'witness_present_encoded' in result.columns
        
        # Check values are numeric
        assert result['witness_present_encoded'].dtype in [np.int64, np.float64]
    
    def test_scaling(self, sample_df):
        """Test feature scaling"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        # Check scaled columns exist
        assert 'age_scaled' in result.columns or 'policy_annual_premium_scaled' in result.columns
    
    def test_missing_value_handling(self):
        """Test missing value imputation"""
        df = pd.DataFrame({
            'months_as_customer': [24, None, 12],
            'age': [35, 45, None],
            'policy_annual_premium': [1200.0, 1500.0, 1000.0],
            'incident_severity': ['Minor Damage', 'Major Damage', 'Trivial Damage'],
            'total_claim_amount': [5000.0, 15000.0, 800.0],
            'injury_claim': [2000.0, 5000.0, 0.0],
            'property_claim': [2000.0, 5000.0, 500.0],
            'vehicle_claim': [1000.0, 5000.0, 300.0],
            'incident_type': ['Single Vehicle', 'Multi-Vehicle', 'Parked Car'],
            'collision_type': ['Front Collision', 'Rear Collision', 'No Collision'],
            'authorities_contacted': ['Police', 'Police', 'None'],
            'witness_present': ['Yes', 'No', 'No'],
            'police_report_available': ['Yes', 'Yes', 'No']
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(df)
        
        # Check no missing values remain
        assert result['months_as_customer'].isna().sum() == 0
        assert result['age'].isna().sum() == 0
    
    def test_feature_groups(self, sample_df):
        """Test feature group organization"""
        preprocessor = DataPreprocessor()
        result = preprocessor.preprocess(sample_df)
        
        groups = preprocessor.get_feature_groups()
        
        assert 'markov_chain' in groups
        assert 'hmm' in groups
        assert 'mdp' in groups
        assert 'metadata' in groups
        
        # Check that groups contain valid columns
        for group_name, columns in groups.items():
            for col in columns:
                assert col in result.columns or col in ['claim_id', '_ingestion_timestamp']