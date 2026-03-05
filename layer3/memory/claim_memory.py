# layer3/memory/claim_memory.py
"""
Long-term memory for claims and decisions
"""

import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

class ClaimMemory:
    """
    Store and retrieve claim history
    """
    
    def __init__(self, storage_path: str = "data/memory/claims.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = {}
        self._load()
    
    def _load(self):
        """Load memory from disk"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                self.memory = json.load(f)
    
    def save(self):
        """Persist memory to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=2, default=str)
    
    def remember(self, claim_id: str, data: Dict):
        """Store claim in memory"""
        self.memory[claim_id] = {
            'data': data,
            'stored_at': datetime.now().isoformat(),
            'access_count': 0
        }
        self.save()
    
    def recall(self, claim_id: str) -> Optional[Dict]:
        """Retrieve claim from memory"""
        if claim_id in self.memory:
            self.memory[claim_id]['access_count'] += 1
            self.memory[claim_id]['last_accessed'] = datetime.now().isoformat()
            return self.memory[claim_id]['data']
        return None
    
    def search_similar(self, claim_features: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar past claims (simplified)"""
        # In production, use embeddings/vector search
        similar = []
        for claim_id, mem in self.memory.items():
            score = self._similarity_score(claim_features, mem['data'])
            similar.append({'claim_id': claim_id, 'score': score, 'data': mem['data']})
        
        similar.sort(key=lambda x: x['score'], reverse=True)
        return similar[:top_k]
    
    def _similarity_score(self, feat1: Dict, feat2: Dict) -> float:
        """Calculate simple similarity"""
        keys = ['red_flag_count', 'severity_score', 'claim_to_premium_ratio']
        score = 0
        for k in keys:
            if k in feat1 and k in feat2:
                score += 1 - abs(feat1[k] - feat2[k]) / max(feat1[k], feat2[k], 1)
        return score / len(keys)
    
    def get_stats(self) -> Dict:
        """Memory statistics"""
        return {
            'total_claims': len(self.memory),
            'storage_size_mb': self.storage_path.stat().st_size / (1024*1024) if self.storage_path.exists() else 0
        }