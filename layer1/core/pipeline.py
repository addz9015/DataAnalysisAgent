# layer1/core/pipeline.py
"""
Main Pipeline Orchestrator
Coordinates all stages of Layer 1 processing
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging.config
from layer1.config.settings import LOGGING_CONFIG, PROCESSED_DATA_DIR

from .intake import DataIntake
from .validation import DataValidator
from .preprocessing import DataPreprocessor
from .feature_store import FeatureStore

logger = logging.getLogger("layer1.pipeline")

class Layer1Pipeline:
    """
    End-to-end Layer 1 pipeline: Intake → Validation → Preprocessing → Feature Store
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Setup logging
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # Initialize components
        self.intake = DataIntake(self.config.get('intake'))
        self.validator = DataValidator(
            strict_mode=self.config.get('strict_mode', False),
            max_missing_ratio=self.config.get('max_missing_ratio', 0.3)
        )
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing'))
        storage_path = self.config.get('storage_path', str(PROCESSED_DATA_DIR))
        self.feature_store = FeatureStore(storage_path)
        
        # Pipeline state
        self.processing_history = []
        self.current_batch_id = None
    
    def process(self, source: Any, export_csv: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute full Layer 1 pipeline
        
        Args:
            source: Input data (file path, DataFrame, etc.)
            export_csv: Whether to export results for Tableau
        
        Returns:
            processed_df: Fully processed DataFrame
            report: Comprehensive processing report
        """
        start_time = datetime.now()
        self.current_batch_id = self.intake._generate_batch_id()
        
        logger.info(f"=== Starting Layer 1 Pipeline | Batch: {self.current_batch_id} ===")
        
        try:
            # Stage 1: Intake
            raw_df = self.intake.receive(source)
            
            # Stage 2: Validation
            valid_df, validation_report = self.validator.validate_batch(raw_df)
            
            if len(valid_df) == 0:
                raise ValueError("No valid records after validation")
            
            # Stage 3: Preprocessing
            processed_df = self.preprocessor.preprocess(valid_df)
            
            # Stage 4: Feature Store
            feature_groups = self.preprocessor.get_feature_groups()
            self.feature_store.store_batch(processed_df, feature_groups)
            self.feature_store.save("feature_store.json")
            
            # Export processed features
            csv_path = None
            if export_csv:
                csv_path = self.feature_store.export_to_csv(processed_df, "processed_features.csv")
            
            # Generate final report
            processing_time = (datetime.now() - start_time).total_seconds()
            report = self._generate_report(
                raw_df, valid_df, processed_df,
                validation_report, processing_time, csv_path
            )
            
            # Log to history
            self.processing_history.append({
                'batch_id': self.current_batch_id,
                'timestamp': start_time.isoformat(),
                'records_processed': len(processed_df),
                'status': 'success'
            })
            
            logger.info(f"=== Pipeline Complete | Processed {len(processed_df)} records in {processing_time:.2f}s ===")
            
            return processed_df, report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.processing_history.append({
                'batch_id': self.current_batch_id,
                'timestamp': start_time.isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    def _generate_report(self, raw_df: pd.DataFrame, valid_df: pd.DataFrame,
                        processed_df: pd.DataFrame, validation_report: Dict,
                        processing_time: float, csv_path: Optional[str]) -> Dict:
        """Generate comprehensive processing report"""
        
        return {
            'batch_id': self.current_batch_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'records': {
                'input': len(raw_df),
                'after_validation': len(valid_df),
                'output': len(processed_df),
                'validation_drop_rate': (len(raw_df) - len(valid_df)) / len(raw_df) * 100
            },
            'features': {
                'input_columns': list(raw_df.columns),
                'output_columns': list(processed_df.columns),
                'total_features': len(processed_df.columns),
                'feature_groups': self.preprocessor.get_feature_groups()
            },
            'validation': validation_report,
            'outputs': {
                'csv_export_path': csv_path,
                'feature_store_path': str(self.feature_store.storage_path / "feature_store.json") if self.feature_store.storage_path else None,
                'feature_store_stats': self.feature_store.get_statistics()
            },
            'markov_state_distribution': processed_df['markov_state'].value_counts().to_dict(),
            'data_quality_score': validation_report['summary']['valid_percentage']
        }
    
    def get_history(self) -> List[Dict]:
        """Get processing history"""
        return self.processing_history
    
    def reset(self):
        """Reset pipeline state"""
        self.processing_history = []
        self.current_batch_id = None
        self.feature_store = FeatureStore(self.config.get('storage_path'))
        logger.info("Pipeline reset")