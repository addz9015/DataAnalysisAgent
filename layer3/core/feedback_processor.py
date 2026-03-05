# layer3/core/feedback_processor.py
"""
Process human feedback and corrections
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("layer3.feedback")

class FeedbackProcessor:
    """
    Handle human-in-the-loop feedback
    """
    
    def __init__(self):
        self.feedback_log = []
        self.corrections = []
        
    def receive_feedback(self,
                        claim_id: str,
                        agent_decision: str,
                        human_decision: str,
                        reason: Optional[str] = None,
                        rater_id: Optional[str] = None) -> Dict:
        """
        Process human correction to agent decision
        """
        feedback = {
            'claim_id': claim_id,
            'timestamp': pd.Timestamp.now(),
            'agent_decision': agent_decision,
            'human_decision': human_decision,
            'agreement': agent_decision == human_decision,
            'reason': reason,
            'rater_id': rater_id,
            'severity': self._assess_severity(agent_decision, human_decision)
        }
        
        self.feedback_log.append(feedback)
        
        if not feedback['agreement']:
            self.corrections.append(feedback)
            logger.warning(
                f"Human correction on {claim_id}: "
                f"agent={agent_decision}, human={human_decision}"
            )
        
        return feedback
    
    def _assess_severity(self, agent: str, human: str) -> str:
        """Assess severity of disagreement"""
        # Critical: Agent approved fraud or denied legitimate
        if (agent in ['approve', 'fast_track'] and human in ['deny', 'deep']) or \
           (agent == 'deny' and human in ['approve', 'fast_track']):
            return 'critical'
        
        # Moderate: Investigation level disagreement
        if set([agent, human]) == set(['standard', 'deep']):
            return 'minor'
        
        # Low: Fast-track vs standard
        return 'low'
    
    def get_disagreement_patterns(self) -> Dict:
        """Find patterns in human-agent disagreements"""
        if not self.corrections:
            return {'status': 'no_corrections'}
        
        df = pd.DataFrame(self.corrections)
        
        patterns = {
            'total_corrections': len(df),
            'by_agent_decision': df['agent_decision'].value_counts().to_dict(),
            'by_human_decision': df['human_decision'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'common_reasons': df['reason'].value_counts().head(5).to_dict() if 'reason' in df else {}
        }
        
        # Agent tends to be too lenient or too strict?
        agent_approves_human_denies = len(df[
            (df['agent_decision'].isin(['approve', 'fast_track'])) & 
            (df['human_decision'].isin(['deny', 'deep']))
        ])
        
        agent_denies_human_approves = len(df[
            (df['agent_decision'] == 'deny') & 
            (df['human_decision'].isin(['approve', 'fast_track']))
        ])
        
        patterns['bias_assessment'] = {
            'agent_too_lenient': agent_approves_human_denies,
            'agent_too_strict': agent_denies_human_approves,
            'assessment': 'too_lenient' if agent_approves_human_denies > agent_denies_human_approves else 'too_strict' if agent_denies_human_approves > agent_approves_human_denies else 'balanced'
        }
        
        return patterns
    
    def generate_retraining_recommendations(self) -> List[Dict]:
        """Recommend claims for retraining based on feedback"""
        if not self.corrections:
            return []
        
        # Priority: Critical disagreements first
        critical = [c for c in self.corrections if c['severity'] == 'critical']
        
        recommendations = []
        for corr in critical[:50]:  # Top 50 critical
            recommendations.append({
                'claim_id': corr['claim_id'],
                'priority': 'high',
                'reason': f"Critical disagreement: agent={corr['agent_decision']}, human={corr['human_decision']}",
                'use_for_retraining': True
            })
        
        return recommendations
    
    def get_feedback_stats(self) -> Dict:
        """Overall feedback statistics"""
        if not self.feedback_log:
            return {'status': 'no_feedback'}
        
        total = len(self.feedback_log)
        agreements = sum(1 for f in self.feedback_log if f['agreement'])
        
        return {
            'total_feedback': total,
            'agreements': agreements,
            'disagreements': total - agreements,
            'agreement_rate': agreements / total,
            'correction_rate': len(self.corrections) / total,
            'severity_breakdown': pd.DataFrame(self.feedback_log)['severity'].value_counts().to_dict() if self.feedback_log else {}
        }