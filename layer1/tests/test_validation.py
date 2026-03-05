# layer1/tests/test_validation.py
"""
Tests for validation module
"""

import pytest
import pandas as pd
import numpy as np

from layer1.core.validation import DataValidator, ClaimSchema
from pydantic import ValidationError

class TestClaimSchema:
    
    def test_valid_claim(self):
        """Test valid claim passes validation"""
        claim = {
            'months_as_customer': 24,
            'age': 35,
            'policy_annual_premium': 1200.0,
            'incident_severity': 'Minor Damage',
            'total_claim_amount': 5000.0,
            'injury_claim': 2000.0,
            'property_claim': 2000.0,
            'vehicle_claim': 1000.0,
            'incident_type': 'Single Vehicle',
            'collision_type': 'Front Collision',
            'authorities_contacted': 'Police',
            'witness_present': 'Yes',
            'police_report_available': 'Yes'
        }
        
        # Should not raise exception
        validated = ClaimSchema(**claim)
        assert validated.age == 35
    
    def test_invalid_age(self):
        """Test age validation"""
        with pytest.raises(ValidationError):
            ClaimSchema(
                months_as_customer=24,
                age=150,  # Invalid: too old
                policy_annual_premium=1200.0,
                incident_severity='Minor Damage',
                total_claim_amount=5000.0,
                injury_claim=2000.0,
                property_claim=2000.0,
                vehicle_claim=1000.0,
                incident_type='Single Vehicle',
                collision_type='Front Collision',
                authorities_contacted='Police',
                witness_present='Yes',
                police_report_available='Yes'
            )
    
    def test_invalid_severity(self):
        """Test incident severity validation"""
        with pytest.raises(ValidationError):
            ClaimSchema(
                months_as_customer=24,
                age=35,
                policy_annual_premium=1200.0,
                incident_severity='Invalid Severity',  # Not in allowed list
                total_claim_amount=5000.0,
                injury_claim=2000.0,
                property_claim=2000.0,
                vehicle_claim=1000.0,
                incident_type='Single Vehicle',
                collision_type='Front Collision',
                authorities_contacted='Police',
                witness_present='Yes',
                police_report_available='Yes'
            )
    
    def test_amount_consistency(self):
        """Test that total claim equals sum of components"""
        with pytest.raises(ValidationError):
            ClaimSchema(
                months_as_customer=24,
                age=35,
                policy_annual_premium=1200.0,
                incident_severity='Minor Damage',
                total_claim_amount=10000.0,  # Doesn't match sum
                injury_claim=2000.0,
                property_claim=2000.0,
                vehicle_claim=1000.0,  # Sum = 5000, not 10000
                incident_type='Single Vehicle',
                collision_type='Front Collision',
                authorities_contacted='Police',
                witness_present='Yes',
                police_report_available='Yes'
            )

class TestDataValidator:
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'months_as_customer': [24, 36, 12, 48],
            'age': [35, 45, 28, 55],
            'policy_annual_premium': [1200.0, 1500.0, 1000.0, 2000.0],
            'incident_severity': ['Minor Damage', 'Major Damage', 'Trivial Damage', 'Total Loss'],
            'total_claim_amount': [5000.0, 15000.0, 800.0, 25000.0],
            'injury_claim': [2000.0, 5000.0, 0.0, 10000.0],
            'property_claim': [2000.0, 5000.0, 500.0, 10000.0],
            'vehicle_claim': [1000.0, 5000.0, 300.0, 5000.0],
            'incident_type': ['Single Vehicle', 'Multi-Vehicle', 'Parked Car', 'Vehicle Theft'],
            'collision_type': ['Front Collision', 'Rear Collision', 'No Collision', 'Side Collision'],
            'authorities_contacted': ['Police', 'Police', 'None', 'Fire'],
            'witness_present': ['Yes', 'No', 'No', 'Yes'],
            'police_report_available': ['Yes', 'Yes', 'No', 'Unknown']
        })
    
    def test_valid_batch(self, sample_df):
        """Test validation of valid batch"""
        validator = DataValidator()
        valid_df, report = validator.validate_batch(sample_df)
        
        assert len(valid_df) == 4
        assert report['summary']['valid_records'] == 4
        assert report['summary']['invalid_records'] == 0
    
    def test_invalid_records_filtered(self):
        """Test that invalid records are filtered out"""
        df = pd.DataFrame({
            'months_as_customer': [24, -5, 36],  # -5 is invalid
            'age': [35, 45, 150],  # 150 is invalid
            'policy_annual_premium': [1200.0, 1500.0, -100.0],  # negative is invalid
            'incident_severity': ['Minor Damage', 'Invalid', 'Major Damage'],
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
        
        validator = DataValidator()
        valid_df, report = validator.validate_batch(df)
        
        assert len(valid_df) == 1  # Only first record is valid
        assert report['summary']['invalid_records'] == 2
    
    def test_error_tracking(self, sample_df):
        """Test that errors are tracked"""
        # Add some invalid records
        df = sample_df.copy()
        df.loc[0, 'age'] = 150  # Invalid
        
        validator = DataValidator()
        validator.validate_batch(df)
        
        assert len(validator.error_patterns) > 0
    
    def test_report_generation(self, sample_df):
        """Test that comprehensive report is generated"""
        validator = DataValidator()
        valid_df, report = validator.validate_batch(sample_df)
        
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'missing_value_stats' in report
        assert 'recommendations' in report
    
    def test_missing_value_stats(self):
        """Test missing value statistics"""
        df = pd.DataFrame({
            'months_as_customer': [24, 36, None],
            'age': [35, None, 28],
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
        
        validator = DataValidator()
        valid_df, report = validator.validate_batch(df)
        
        assert 'months_as_customer' in report['missing_value_stats']
        assert report['missing_value_stats']['months_as_customer']['count'] == 1