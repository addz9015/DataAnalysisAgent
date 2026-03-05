# layer3/memory/performance_tracker.py
"""
Track agent performance over time
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from collections import deque

class PerformanceTracker:
    """
    Track and visualize agent performance metrics
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.decisions = deque(maxlen=window_size)
        self.metrics_history = []
        
    def record(self, 
               timestamp: datetime,
               decision: str,
               fraud_prob: float,
               confidence: float,
               processing_time_ms: float,
               outcome: str = None):
        """Record a decision"""
        self.decisions.append({
            'timestamp': timestamp,
            'decision': decision,
            'fraud_prob': fraud_prob,
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'outcome': outcome
        })
    
    def get_current_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if len(self.decisions) < 10:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.decisions)
        
        return {
            'total_decisions': len(df),
            'avg_confidence': df['confidence'].mean(),
            'avg_processing_time_ms': df['processing_time_ms'].mean(),
            'decision_distribution': df['decision'].value_counts().to_dict(),
            'fraud_prob_distribution': {
                'mean': df['fraud_prob'].mean(),
                'std': df['fraud_prob'].std()
            },
            'outcome_accuracy': self._calculate_accuracy(df) if 'outcome' in df else None
        }
    
    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate accuracy if outcomes available"""
        # Simplified - would need actual outcome mapping
        return None
    
    def get_trends(self, periods: int = 10) -> List[Dict]:
        """Get performance trends over time"""
        if len(self.decisions) < periods * 10:
            return []
        
        df = pd.DataFrame(self.decisions)
        df['period'] = pd.cut(range(len(df)), periods)
        
        trends = []
        for period, group in df.groupby('period'):
            trends.append({
                'period': int(period),
                'avg_confidence': group['confidence'].mean(),
                'avg_fraud_prob': group['fraud_prob'].mean(),
                'decision_count': len(group)
            })
        
        return trends