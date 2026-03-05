# layer1/tests/test_pipeline.py
"""
Unit tests for Layer 1 pipeline
"""

import pytest
import pandas as pd
import numpy as np
from layer1.core.pipeline import Layer1Pipeline

@pytest.fixture
def sample_data():
    """Create sample claim data for testing"""
    return pd.DataFrame({
        'months_as_customer': [24, 36, 12],
        'age': [35, 45, 28],
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
        'police_report_available': ['Yes', 'Yes', 'No'],
        'claim_id': ['C001', 'C002', 'C003']
    })

def test_full_pipeline(sample_data):
    """Test complete pipeline execution"""
    pipeline = Layer1Pipeline()
    processed_df, report = pipeline.process(sample_data, export_csv=False)
    
    assert len(processed_df) == 3
    assert 'markov_state' in processed_df.columns
    assert 'fraud_probability' not in processed_df.columns  # Layer 2 adds this
    assert report['records']['output'] == 3

def test_markov_state_derivation(sample_data):
    """Test that Markov states are correctly assigned"""
    pipeline = Layer1Pipeline()
    processed_df, _ = pipeline.process(sample_data, export_csv=False)
    
    # Check states exist
    assert all(state in processed_df['markov_state'].values 
               for state in ['Fast_Track', 'Standard_Investigation'])
    
    # Check encoding
    assert processed_df['markov_state_encoded'].dtype == np.int64