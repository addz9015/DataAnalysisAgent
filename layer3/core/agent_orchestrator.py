"""
Main Agent: Coordinates all Layer 3 components with Hybrid Explainer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from .reasoning_engine import ReasoningEngine, SituationAnalysis
from .action_selector import ActionSelector, AgentDecision
from .explanation_generator import ExplanationGenerator
from .hybrid_explainer import HybridExplainer  # NEW
from ..llm.explainer_llm import LLMExplainer  # NEW

logger = logging.getLogger("layer3.orchestrator")

class StochClaimAgent:
    """
    Autonomous agent for insurance fraud detection with Hybrid Explainer
    """
    
    def __init__(self, 
                 risk_tolerance: str = 'balanced',
                 use_llm: bool = True,
                 llm_provider: str = 'groq',
                 llm_threshold: float = 0.6,  # Only use LLM for fraud_prob > 60%
                 auto_approve_threshold: float = 0.15,
                 auto_deny_threshold: float = 0.85):
        
        self.reasoning = ReasoningEngine(risk_tolerance=risk_tolerance)
        self.selector = ActionSelector(
            auto_approve_threshold=auto_approve_threshold,
            auto_deny_threshold=auto_deny_threshold
        )
        
        # NEW: Hybrid explainer (templates + LLM)
        if use_llm:
            base_llm = LLMExplainer(provider=llm_provider)
            self.explainer = HybridExplainer(
                llm_explainer=base_llm,
                llm_threshold=llm_threshold
            )
            logger.info(f"Initialized Hybrid Explainer (LLM threshold: {llm_threshold})")
        else:
            self.explainer = ExplanationGenerator(use_llm=False)
        
        self.decision_history = []
        self.performance_tracker = []
        
    def process_claim(self, claim_row: pd.Series, 
                      historical_stats: Optional[Dict] = None) -> Dict:
        """
        Process single claim through full agent pipeline with hybrid explanation
        """
        claim_id = str(claim_row.get('claim_id', 'unknown'))
        logger.info(f"Processing claim {claim_id}")
        
        # Step 1: Reasoning - analyze situation
        analysis = self.reasoning.analyze(claim_row)
        
        # Step 2: Evaluate all options
        options = self.reasoning.evaluate_options(analysis)
        
        # Step 3: Detect anomalies
        anomalies = self.reasoning.detect_anomalies(
            claim_row, 
            historical_stats or {}
        )
        
        # Step 4: Make decision
        decision = self.selector.decide(analysis, options, anomalies)
        
        # Step 5: Generate explanations using HYBRID approach
        explanations = self._generate_explanations_hybrid(decision, analysis, claim_row)
        
        # Step 6: Log decision
        self._log_decision(claim_id, decision, analysis)
        
        return {
            'claim_id': claim_id,
            'decision': decision,
            'analysis': analysis,
            'explanations': explanations,
            'anomalies_detected': anomalies,
            'explanation_source': explanations.get('source', 'unknown')  # Track if LLM or template
        }
    
    def _generate_explanations_hybrid(self, decision: AgentDecision, 
                                      analysis: SituationAnalysis,
                                      claim_row: pd.Series) -> Dict[str, str]:
        """
        Generate explanations using hybrid approach (template + LLM)
        """
        # Prepare data for explainer
        explainer_data = {
            'claim_id': decision.claim_id,
            'agent_decision': decision.selected_action,
            'fraud_probability': analysis.fraud_probability,
            'confidence': decision.confidence,
            'risk_score': decision.risk_score,
            'red_flag_count': analysis.key_evidence.get('red_flags', 0),
            'claim_to_premium_ratio': analysis.key_evidence.get('claim_premium_ratio', 0),
            'severity': analysis.key_evidence.get('severity', 'Unknown'),
            'expected_fraud_loss': analysis.expected_fraud_loss,
            'investigation_cost': analysis.investigation_cost.get(decision.selected_action, 0),
            'reasoning': decision.reasoning
        }
        
        # Use hybrid explainer (template or LLM based on complexity)
        detailed_explanation = self.explainer.explain(explainer_data)
        
        # Determine source
        source = 'llm' if analysis.fraud_probability > self.explainer.llm_threshold else 'template'
        
        return {
            'summary': self._generate_summary(decision, analysis),
            'detailed': detailed_explanation,
            'technical': self._generate_technical(decision, analysis),
            'structured': self._generate_structured(decision, analysis),
            'source': source,  # 'llm' or 'template'
            'template_used': None if source == 'llm' else decision.selected_action
        }
    
    def _generate_summary(self, decision: AgentDecision, analysis: SituationAnalysis) -> str:
        """One-line summary"""
        action_desc = {
            'approve': 'approved',
            'fast_track': 'fast-tracked',
            'standard': 'sent for standard investigation',
            'deep': 'flagged for deep investigation',
            'deny': 'denied'
        }
        
        return (
            f"Claim {decision.claim_id} {action_desc.get(decision.selected_action, 'processed')} "
            f"with {decision.confidence:.0%} confidence. "
            f"Risk score: {decision.risk_score}/100."
        )
    
    def _generate_technical(self, decision: AgentDecision, analysis: SituationAnalysis) -> str:
        """Technical JSON explanation"""
        import json
        return json.dumps({
            'claim_id': decision.claim_id,
            'decision': decision.selected_action,
            'confidence': decision.confidence,
            'fraud_probability': analysis.fraud_probability,
            'uncertainty': analysis.uncertainty,
            'expected_value_by_action': analysis.investigation_cost,
            'risk_score': decision.risk_score,
            'sla_hours': decision.sla_hours,
            'requires_human_review': decision.requires_human_review
        }, indent=2)
    
    def _generate_structured(self, decision: AgentDecision, analysis: SituationAnalysis) -> Dict:
        """Structured dict for API/frontend"""
        return {
            'claim_id': decision.claim_id,
            'decision': {
                'action': decision.selected_action,
                'confidence': decision.confidence,
                'risk_score': decision.risk_score
            },
            'rationale': {
                'fraud_probability': analysis.fraud_probability,
                'key_factors': analysis.key_evidence,
                'reasoning_text': decision.reasoning
            },
            'next_steps': {
                'sla_hours': decision.sla_hours,
                'investigation_depth': decision.investigation_depth,
                'human_review_required': decision.requires_human_review
            }
        }
    
    def process_batch(self, df: pd.DataFrame,
                      historical_stats: Optional[Dict] = None,
                      show_progress: bool = True) -> pd.DataFrame:
        """
        Process multiple claims with progress tracking
        """
        results = []
        llm_count = 0
        template_count = 0
        
        total = len(df)
        
        for i, (_, row) in enumerate(df.iterrows()):
            result = self.process_claim(row, historical_stats)
            results.append(self._flatten_result(result))
            
            # Track explanation source
            if result['explanations'].get('source') == 'llm':
                llm_count += 1
            else:
                template_count += 1
            
            # Progress
            if show_progress and (i + 1) % 10 == 0:
                print(f"✓ Processed {i+1}/{total} (LLM: {llm_count}, Template: {template_count})")
        
        # Summary
        print(f"\n{'='*50}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total: {total}")
        print(f"LLM explanations: {llm_count} ({llm_count/total*100:.1f}%)")
        print(f"Template explanations: {template_count} ({template_count/total*100:.1f}%)")
        print(f"Estimated API cost saved: {template_count * 0.002:.2f}")  # Assuming $0.002 per call
        
        return pd.DataFrame(results)
    
    def _flatten_result(self, result: Dict) -> Dict:
        """Flatten nested result for DataFrame output"""
        decision = result['decision']
        analysis = result['analysis']
        explanations = result['explanations']
        
        return {
            'claim_id': decision.claim_id,
            'agent_action': decision.selected_action,
            'agent_confidence': decision.confidence,
            'agent_reasoning': decision.reasoning,
            'risk_score': decision.risk_score,
            'requires_human_review': decision.requires_human_review,
            'sla_hours': decision.sla_hours,
            'investigation_depth': decision.investigation_depth,
            'fraud_probability': analysis.fraud_probability,
            'expected_fraud_loss': analysis.expected_fraud_loss,
            'explanation_summary': result['explanations']['summary'],
            'anomalies': '|'.join(result['anomalies_detected'])
        }
    
    def _log_decision(self, claim_id: str, decision: AgentDecision, 
                      analysis: SituationAnalysis):
        """Log decision for learning"""
        self.decision_history.append({
            'claim_id': claim_id,
            'timestamp': pd.Timestamp.now(),
            'action': decision.selected_action,
            'fraud_probability': analysis.fraud_probability,
            'risk_score': decision.risk_score
        })
    
    def provide_feedback(self, claim_id: str, actual_outcome: str):
        """
        Learn from actual outcomes (fraud confirmed or not)
        """
        # Find decision in history
        decision = next(
            (d for d in self.decision_history if d['claim_id'] == claim_id),
            None
        )
        
        if decision:
            self.performance_tracker.append({
                'claim_id': claim_id,
                'predicted_action': decision['action'],
                'predicted_fraud_prob': decision['fraud_probability'],
                'actual_outcome': actual_outcome,
                'correct': (decision['action'] == 'deny' and actual_outcome == 'fraud') or
                          (decision['action'] == 'approve' and actual_outcome == 'legitimate')
            })
            
            # Adapt thresholds if enough data
            if len(self.performance_tracker) >= 100:
                self._adapt_if_needed()
    
    def _adapt_if_needed(self):
        """Adapt agent based on performance"""
        perf_df = pd.DataFrame(self.performance_tracker[-100:])
        self.selector.adapt_thresholds(perf_df)
    
    def get_performance_report(self) -> Dict:
        """Get agent performance statistics"""
        if not self.performance_tracker:
            return {'status': 'No feedback yet'}
        
        df = pd.DataFrame(self.performance_tracker)
        
        return {
            'total_decisions': len(df),
            'accuracy': df['correct'].mean(),
            'false_positives': len(df[(df['predicted_action'] == 'deny') & 
                                      (df['actual_outcome'] == 'legitimate')]),
            'false_negatives': len(df[(df['predicted_action'] == 'approve') & 
                                      (df['actual_outcome'] == 'fraud')]),
            'avg_fraud_probability': df['predicted_fraud_prob'].mean()
        }