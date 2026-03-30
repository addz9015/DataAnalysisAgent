# layer1/core/validation.py
"""
Stage 2: Schema Validation
Validates data against predefined schemas using Pydantic
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, ValidationError, Field, field_validator, model_validator
from datetime import datetime

logger = logging.getLogger("layer1.validation")

# Pydantic model for individual claim validation
class ClaimSchema(BaseModel):
    """Validation schema for insurance claims"""
    
    # Required fields
    months_as_customer: int = Field(..., ge=0, le=600)
    age: int = Field(..., ge=16, le=100)
    policy_annual_premium: float = Field(..., gt=0)
    incident_severity: str
    total_claim_amount: float = Field(..., ge=0)
    injury_claim: float = Field(..., ge=0)
    property_claim: float = Field(..., ge=0)
    vehicle_claim: float = Field(..., ge=0)
    incident_type: str
    collision_type: Optional[str] = None
    authorities_contacted: Optional[str] = None
    witness_present: Optional[str] = None
    witnesses: Optional[int] = Field(default=None, ge=0)
    police_report_available: Optional[str] = None
    
    # Optional fields with defaults
    claim_id: Optional[str] = None
    incident_date: Optional[str] = None
    fraud_reported: Optional[str] = None
    claim_status: Optional[str] = None
    settlement_amount: Optional[float] = None
    
    # Validators for categorical fields
    @field_validator('incident_severity')
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']
        if v not in allowed:
            raise ValueError(f'incident_severity must be one of {allowed}')
        return v
    
    @field_validator('incident_type')
    @classmethod
    def validate_incident_type(cls, v: str) -> str:
        allowed = {
            'Single Vehicle': 'Single Vehicle',
            'Single Vehicle Collision': 'Single Vehicle',
            'Multi-Vehicle': 'Multi-Vehicle',
            'Multi-vehicle Collision': 'Multi-Vehicle',
            'Vehicle Theft': 'Vehicle Theft',
            'Parked Car': 'Parked Car'
        }
        value = str(v).strip()
        if value not in allowed:
            raise ValueError(f'incident_type must be one of {list(allowed.keys())}')
        return allowed[value]

    @field_validator('collision_type')
    @classmethod
    def validate_collision_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = [
            'Front Collision', 'Rear Collision', 'Side Collision', 'No Collision'
        ]
        if v not in allowed:
            raise ValueError(f'collision_type must be one of {allowed}')
        return v
    
    @field_validator('authorities_contacted')
    @classmethod
    def validate_authorities(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = ['Police', 'Fire', 'Ambulance', 'None', 'Other']
        if v not in allowed:
            raise ValueError(f'authorities_contacted must be one of {allowed}')
        return v

    @field_validator('witness_present')
    @classmethod
    def validate_witness_present(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if str(v).strip().lower() not in ['yes', 'no', 'unknown', 'y', 'n', '1', '0']:
            raise ValueError('witness_present must be Yes/No/Unknown')
        return v

    @model_validator(mode='before')
    @classmethod
    def infer_witness_count(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if data.get('witnesses') is not None:
            return data

        normalized_data = dict(data)
        witness_present = normalized_data.get('witness_present')
        if witness_present is None:
            normalized_data['witnesses'] = 0
            return normalized_data

        normalized = str(witness_present).strip().lower()
        if normalized in ['yes', 'y', '1', 'true']:
            normalized_data['witnesses'] = 1
        elif normalized in ['no', 'n', '0', 'false']:
            normalized_data['witnesses'] = 0
        else:
            normalized_data['witnesses'] = 0
        return normalized_data
    
    @field_validator('police_report_available')
    @classmethod
    def validate_binary(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Allow flexibility in binary representations
        if str(v).strip().lower() not in ['yes', 'no', 'unknown', 'y', 'n', '1', '0']:
            raise ValueError(f'Must be Yes/No/Unknown')
        return v
    
    @model_validator(mode='after')
    def validate_amount_consistency(self):
        """Check total against components after all component amounts are available."""
        total = self.total_claim_amount or 0
        injury = self.injury_claim or 0
        property_c = self.property_claim or 0
        vehicle = self.vehicle_claim or 0

        expected = injury + property_c + vehicle

        # Skip validation if total or components are unavailable.
        if total == 0 or expected == 0:
            return self

        # Allow 10% tolerance for rounding/errors.
        if abs(total - expected) > (expected * 0.1):
            raise ValueError(f'total_claim_amount ({total}) != sum of components ({expected})')
        return self


class DataValidator:
    """
    Validates batches of claims against schema
    Tracks errors and produces validation reports
    """
    
    def __init__(self, strict_mode: bool = False, max_missing_ratio: float = 0.3):
        self.strict_mode = strict_mode
        self.max_missing_ratio = max_missing_ratio
        self.validation_results = []
        self.error_patterns = {}
        
    def validate_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate entire DataFrame
        
        Returns:
            valid_df: DataFrame with only valid records
            report: Detailed validation report
        """
        logger.info(f"Starting validation of {len(df)} records")
        
        valid_records = []
        invalid_records = []
        
        for idx, row in df.iterrows():
            is_valid, errors = self._validate_single_record(row)
            
            if is_valid:
                valid_records.append(row)
            else:
                invalid_records.append({
                    'index': idx,
                    'errors': errors,
                    'data': row.to_dict()
                })
                self._track_errors(errors)
        
        # Create result DataFrame
        valid_df = pd.DataFrame(valid_records) if valid_records else pd.DataFrame()
        
        # Generate report
        report = self._generate_report(df, valid_df, invalid_records)
        
        logger.info(f"Validation complete: {len(valid_df)} valid, {len(invalid_records)} invalid")
        
        if self.strict_mode and len(invalid_records) > 0:
            logger.warning(f"Strict mode: {len(invalid_records)} invalid records found")
        
        return valid_df, report
    
    def _validate_single_record(self, row: pd.Series) -> Tuple[bool, List[str]]:
        """Validate single record using Pydantic"""
        try:
            # Convert row to dict, handling NaN and '?' values
            record_dict = row.replace({np.nan: None, '?': None}).to_dict()
            
            # Attempt validation
            ClaimSchema(**record_dict)
            return True, []
            
        except ValidationError as e:
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            return False, errors
    
    def _track_errors(self, errors: List[str]):
        """Track error patterns for reporting"""
        for error in errors:
            self.error_patterns[error] = self.error_patterns.get(error, 0) + 1
    
    def _generate_report(self, original_df: pd.DataFrame, 
                        valid_df: pd.DataFrame,
                        invalid_records: List[Dict]) -> Dict:
        """Generate comprehensive validation report"""
        
        total = len(original_df)
        valid_count = len(valid_df)
        invalid_count = len(invalid_records)
        
        # Calculate missing value statistics
        missing_stats = {}
        for col in original_df.columns:
            missing_count = original_df[col].isna().sum()
            missing_stats[col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / total * 100)
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_records': total,
                'valid_records': valid_count,
                'invalid_records': invalid_count,
                'valid_percentage': float(valid_count / total * 100) if total > 0 else 0
            },
            'error_breakdown': dict(sorted(
                self.error_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),  # Top 10 errors
            'missing_value_stats': missing_stats,
            'sample_invalid': invalid_records[:10] if invalid_records else [],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Analyze error patterns
        if 'total_claim_amount' in str(self.error_patterns):
            recommendations.append(
                "Check claim amount calculations - many inconsistencies found"
            )
        
        if any('incident_severity' in err for err in self.error_patterns.keys()):
            recommendations.append(
                "Standardize incident severity values before ingestion"
            )
        
        if not recommendations:
            recommendations.append("No major issues detected")
        
        return recommendations
    
    def save_report(self, report: Dict, filepath: str):
        """Save validation report to JSON"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {filepath}")