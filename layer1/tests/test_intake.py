# layer1/tests/test_intake.py
"""
Tests for data intake module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from layer1.core.intake import DataIntake

@pytest.fixture
def sample_csv():
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("months_as_customer,age,policy_annual_premium\n")
        f.write("24,35,1200.0\n")
        f.write("36,45,1500.0\n")
        f.write("12,28,1000.0\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()

@pytest.fixture
def sample_json():
    """Create temporary JSON file for testing"""
    data = [
        {"months_as_customer": 24, "age": 35, "premium": 1200.0},
        {"months_as_customer": 36, "age": 45, "premium": 1500.0}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    Path(temp_path).unlink()

class TestDataIntake:
    
    def test_csv_intake(self, sample_csv):
        """Test CSV file intake"""
        intake = DataIntake()
        df = intake.receive(sample_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'months_as_customer' in df.columns
    
    def test_json_intake(self, sample_json):
        """Test JSON file intake"""
        intake = DataIntake()
        df = intake.receive(sample_json)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_dict_intake(self):
        """Test single dictionary intake"""
        intake = DataIntake()
        record = {
            "months_as_customer": 24,
            "age": 35,
            "policy_annual_premium": 1200.0
        }
        
        df = intake.receive(record)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['age'].iloc[0] == 35
    
    def test_list_intake(self):
        """Test list of dictionaries intake"""
        intake = DataIntake()
        records = [
            {"months_as_customer": 24, "age": 35},
            {"months_as_customer": 36, "age": 45}
        ]
        
        df = intake.receive(records)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_dataframe_intake(self):
        """Test DataFrame pass-through"""
        intake = DataIntake()
        original_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        df = intake.receive(original_df)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        pd.testing.assert_frame_equal(df, original_df)
    
    def test_column_standardization(self):
        """Test that column names are standardized"""
        intake = DataIntake()
        df = pd.DataFrame({
            'Column One': [1, 2],
            'Column Two': ['a', 'b'],
            'COLUMN THREE': [1.0, 2.0]
        })
        
        result = intake.receive(df)
        
        assert 'column_one' in result.columns
        assert 'column_two' in result.columns
        assert 'column_three' in result.columns
    
    def test_metadata_addition(self):
        """Test that metadata columns are added"""
        intake = DataIntake()
        df = pd.DataFrame({'test': [1, 2, 3]})
        
        result = intake.receive(df)
        
        assert '_ingestion_timestamp' in result.columns
        assert '_source_type' in result.columns
        assert '_batch_id' in result.columns
    
    def test_unsupported_format(self):
        """Test error on unsupported format"""
        intake = DataIntake()
        
        with pytest.raises(ValueError, match="Unsupported source type"):
            intake.receive(12345)  # Invalid type
    
    def test_stats_tracking(self):
        """Test that statistics are tracked"""
        intake = DataIntake()
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        
        intake.receive(df)
        stats = intake.get_stats()
        
        assert stats['total_received'] == 5
        assert stats['total_processed'] == 5
    
    def test_ingestion_log(self):
        """Test that ingestion is logged"""
        intake = DataIntake()
        df = pd.DataFrame({'col': [1]})
        
        intake.receive(df)
        
        assert len(intake.ingestion_log) == 1
        assert intake.ingestion_log[0]['status'] == 'success'