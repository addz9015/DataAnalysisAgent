# layer1/core/intake.py
"""
Stage 1: Data Intake
Receives data from multiple sources and converts to standard DataFrame format
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Union, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("layer1.intake")

class DataIntake:
    """
    Handles data ingestion from various sources (CSV, JSON, API, DataFrame)
    """
    
    SUPPORTED_FORMATS = ['.csv', '.json', '.parquet', '.xlsx']
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ingestion_log = []
        self.stats = {
            "total_received": 0,
            "total_processed": 0,
            "errors": []
        }
    
    def receive(self, source: Union[str, Path, Dict, List[Dict], pd.DataFrame], 
                source_type: Optional[str] = None) -> pd.DataFrame:
        """
        Main entry point for data intake
        
        Args:
            source: Input data (file path, dict, or DataFrame)
            source_type: Optional hint ('csv', 'json', 'api', 'dataframe')
        
        Returns:
            Standardized pandas DataFrame
        """
        timestamp = datetime.now()
        logger.info(f"Starting data intake from source type: {type(source)}")
        
        try:
            # Determine source type and route accordingly
            if isinstance(source, (str, Path)):
                df = self._read_file(source)
                detected_type = "file"
            
            elif isinstance(source, dict):
                df = pd.DataFrame([source])
                detected_type = "single_record"
            
            elif isinstance(source, list) and len(source) > 0 and isinstance(source[0], dict):
                df = pd.DataFrame(source)
                detected_type = "batch_records"
            
            elif isinstance(source, pd.DataFrame):
                df = source.copy()
                detected_type = "dataframe"
            
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Add metadata columns
            df['_ingestion_timestamp'] = timestamp
            df['_source_type'] = source_type or detected_type
            df['_batch_id'] = self._generate_batch_id()
            
            # Ensure claim_id exists
            if 'claim_id' not in df.columns:
                if 'policy_number' in df.columns:
                    df['claim_id'] = df['policy_number'].astype(str)
                else:
                    import uuid
                    df['claim_id'] = [f"ID_{uuid.uuid4().hex[:8]}" for _ in range(len(df))]
            
            # Update stats
            self.stats["total_received"] += len(df)
            self.stats["total_processed"] += len(df)
            
            # Log ingestion
            self._log_ingestion(timestamp, detected_type, len(df), "success")
            
            logger.info(f"Successfully ingested {len(df)} records")
            return df
            
        except Exception as e:
            error_msg = f"Intake failed: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append({
                "timestamp": timestamp,
                "error": str(e),
                "source": str(source)[:100]
            })
            self._log_ingestion(timestamp, source_type, 0, "failed", str(e))
            raise
    
    def _read_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Read data from file based on extension"""
        filepath = Path(filepath)
        extension = filepath.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}. "
                           f"Supported: {self.SUPPORTED_FORMATS}")
        
        readers = {
            '.csv': lambda p: pd.read_csv(p),
            '.json': lambda p: pd.read_json(p),
            '.parquet': lambda p: pd.read_parquet(p),
            '.xlsx': lambda p: pd.read_excel(p)
        }
        
        logger.debug(f"Reading file: {filepath}")
        return readers[extension](filepath)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names (lowercase, replace spaces with underscores)"""
        df = df.copy()
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        return df
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch identifier"""
        import uuid
        return f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _log_ingestion(self, timestamp: datetime, source_type: str, 
                       record_count: int, status: str, error: Optional[str] = None):
        """Log ingestion event"""
        self.ingestion_log.append({
            "timestamp": timestamp,
            "source_type": source_type,
            "record_count": record_count,
            "status": status,
            "error": error
        })
    
    def get_stats(self) -> Dict:
        """Return ingestion statistics"""
        return {
            **self.stats,
            "recent_logs": self.ingestion_log[-10:]  # Last 10 entries
        }
    
    def save_log(self, filepath: Optional[str] = None):
        """Save ingestion log to file"""
        import json
        
        filepath = filepath or f"ingestion_log_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filepath, 'w') as f:
            json.dump({
                "stats": self.stats,
                "logs": self.ingestion_log
            }, f, indent=2, default=str)
        
        logger.info(f"Ingestion log saved to: {filepath}")