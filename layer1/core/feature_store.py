# layer1/core/feature_store.py
"""
Stage 4: Feature Store
Organizes and stores features for efficient access by downstream layers
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("layer1.feature_store")

class FeatureStore:
    """
    Simple feature store for organizing and retrieving processed features
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.features = {}
        self.metadata = {}
        self.version = "1.0"
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store(self, claim_id: str, features: Dict[str, Any], 
              feature_group: str = "default"):
        """Store features for a specific claim"""
        if feature_group not in self.features:
            self.features[feature_group] = {}
        
        self.features[feature_group][claim_id] = {
            'features': features,
            'stored_at': datetime.now().isoformat(),
            'version': self.version
        }
    
    def retrieve(self, claim_id: str, feature_group: str = "default") -> Optional[Dict]:
        """Retrieve features for a specific claim"""
        if feature_group in self.features and claim_id in self.features[feature_group]:
            return self.features[feature_group][claim_id]['features']
        return None
    
    def store_batch(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]):
        """Store entire DataFrame organized by feature groups"""
        for group_name, columns in feature_groups.items():
            # Filter columns that exist in dataframe
            available_cols = [col for col in columns if col in df.columns]
            
            if available_cols:
                group_df = df[['claim_id'] + available_cols].copy()
                
                # Store as dictionary, handling NaN values for JSON
                for _, row in group_df.iterrows():
                    claim_id = str(row['claim_id'])
                    # Replace NaN with None for valid JSON serialization
                    features = row[available_cols].replace({np.nan: None}).to_dict()
                    self.store(claim_id, features, group_name)
        
        self.metadata = {
            'stored_at': datetime.now().isoformat(),
            'total_claims': len(df),
            'feature_groups': list(feature_groups.keys()),
            'versions': {group: self.version for group in feature_groups.keys()}
        }
        
        logger.info(f"Stored features for {len(df)} claims in {len(feature_groups)} groups")
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = "processed_features.csv"):
        """Export processed data to CSV"""
        if self.storage_path:
            filepath = self.storage_path / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Exported features to: {filepath}")
            return str(filepath)
        else:
            raise ValueError("Storage path not set")
    
    def get_statistics(self) -> Dict:
        """Get feature store statistics"""
        stats = {
            'metadata': self.metadata,
            'group_counts': {
                group: len(claims) 
                for group, claims in self.features.items()
            }
        }
        return stats
    
    def save(self, filename: str = "feature_store.json"):
        """Persist feature store to disk"""
        if not self.storage_path:
            raise ValueError("Storage path not set")
        
        filepath = self.storage_path / filename
        with open(filepath, 'w') as f:
            json.dump({
                'features': self.features,
                'metadata': self.metadata,
                'version': self.version
            }, f, indent=2, default=str)
        
        logger.info(f"Feature store saved to: {filepath}")