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
                 auto_deny_threshold: float = 0.98):
        
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

        # Keep API outputs consistent with low-risk sanity override decisions.
        if decision.sanity_override:
            adjusted_prob = min(float(analysis.fraud_probability), 0.20)
            claim_amount = float(claim_row.get('total_claim_amount', 0) or 0)
            analysis.fraud_probability = adjusted_prob
            analysis.expected_fraud_loss = adjusted_prob * claim_amount * 0.8
        
        # Step 5: Generate explanations using HYBRID approach
        explanations = self._generate_explanations_hybrid(decision, analysis, claim_row)
        human_review_note = self._build_human_review_note(decision, analysis, anomalies, claim_row)
        
        # Step 6: Log decision
        self._log_decision(claim_id, decision, analysis)
        
        return {
            'claim_id': claim_id,
            'decision': decision,
            'analysis': analysis,
            'explanations': explanations,
            'anomalies_detected': anomalies,
            'human_review_note': human_review_note,
            'explanation_source': explanations.get('source', 'unknown')  # Track if LLM or template
        }

    def _build_human_review_note(self,
                                 decision: AgentDecision,
                                 analysis: SituationAnalysis,
                                 anomalies: List[str],
                                 claim_row: Optional[pd.Series] = None) -> str:
        """Create a claim-specific investigator briefing from the full claim record."""
        if not decision.requires_human_review:
            return ""

        def _safe_text(value, default: str = "") -> str:
            if value is None:
                return default
            text = str(value).strip()
            if text.lower() in {"", "nan", "none", "null", "<na>"}:
                return default
            return text

        def _safe_float(value, default: float = 0.0) -> float:
            if value is None:
                return default
            text = _safe_text(value, "")
            if text == "" or text.lower() in {"unknown", "?"}:
                return default
            try:
                return float(text)
            except (TypeError, ValueError):
                return default

        def _safe_int(value, default: int = 0) -> int:
            try:
                return int(_safe_float(value, float(default)))
            except (TypeError, ValueError):
                return default

        ev  = analysis.key_evidence
        row = claim_row if claim_row is not None else pd.Series(dtype=object)
        fraud_pct = analysis.fraud_probability * 100

        # Pull claim-specific fields.
        incident_type   = _safe_text(row.get('incident_type', ev.get('severity', 'Unknown')), 'Unknown')
        collision_type  = _safe_text(row.get('collision_type', '?'), '?')
        severity        = _safe_text(row.get('incident_severity', ev.get('severity', 'Unknown')), 'Unknown')
        authorities     = _safe_text(row.get('authorities_contacted', 'Unknown'), 'Unknown')
        auto_make       = _safe_text(row.get('auto_make', ''), '')
        auto_model      = _safe_text(row.get('auto_model', ''), '')
        auto_year       = _safe_text(row.get('auto_year', ''), '')
        bodily_injuries = _safe_int(row.get('bodily_injuries', 0), default=0)
        witnesses       = _safe_int(row.get('witnesses', row.get('witness_present', 0)), default=0)
        total_claim     = _safe_float(
            row.get('total_claim_amount', analysis.expected_fraud_loss / max(analysis.fraud_probability, 0.01)),
            default=0.0,
        )
        injury_claim    = _safe_float(row.get('injury_claim', 0), default=0.0)
        property_claim  = _safe_float(row.get('property_claim', 0), default=0.0)
        vehicle_claim   = _safe_float(row.get('vehicle_claim', 0), default=0.0)
        hour_raw        = row.get('incident_hour_of_the_day', None)
        hour_val        = _safe_int(hour_raw, default=-1)
        hour            = hour_val if 0 <= hour_val <= 23 else None
        months_cust     = _safe_int(row.get('months_as_customer', 0), default=0)
        ratio           = _safe_float(
            ev.get('claim_premium_ratio', row.get('claim_to_premium_ratio', 0)),
            default=0.0,
        )
        no_witness      = str(ev.get('witness_present', row.get('witness_present', 'yes'))).upper() in ('NO', 'N', 'FALSE', '0')
        no_police       = str(ev.get('police_report',   row.get('police_report_available', 'yes'))).upper() in ('NO', 'N', 'FALSE', '0')
        red_flags       = _safe_int(ev.get('red_flags', row.get('red_flag_count', 0)), default=0)

        # Risk label.
        if fraud_pct >= 90:
            risk_label = "EXTREME"
        elif fraud_pct >= 70:
            risk_label = "HIGH"
        else:
            risk_label = "ELEVATED"

        # Vehicle description (only when data is present).
        vehicle_desc = " ".join(filter(None, [str(auto_year) if auto_year else '', auto_make, auto_model])).strip()

        # Incident summary sentence.
        time_desc = ""
        if hour is not None:
            h = hour
            if 0 <= h < 6:
                time_desc = " in the early hours of the morning (high-risk time window)"
            elif 22 <= h <= 23:
                time_desc = " late at night (high-risk time window)"
        collision_desc = f" ({collision_type} collision)" if collision_type not in ('?', '', 'Unknown', 'nan') else ""
        incident_summary = (
            f"A {severity.lower() or 'reported'} {incident_type.lower()}{collision_desc} "
            f"was submitted{time_desc}, involving "
            + (f"a {vehicle_desc}" if vehicle_desc else "the insured vehicle")
            + (f", with {bodily_injuries} bodily injur{'y' if bodily_injuries == 1 else 'ies'} reported" if bodily_injuries else "")
            + "."
        )

        # Claim breakdown.
        breakdown_lines = []
        if injury_claim:   breakdown_lines.append(f"  Injury        : ${injury_claim:>10,.0f}")
        if property_claim: breakdown_lines.append(f"  Property      : ${property_claim:>10,.0f}")
        if vehicle_claim:  breakdown_lines.append(f"  Vehicle       : ${vehicle_claim:>10,.0f}")
        if breakdown_lines:
            breakdown_lines.insert(0, f"  Total claimed : ${total_claim:>10,.0f}")
        claim_breakdown = ("\n\nCLAIM BREAKDOWN\n" + "\n".join(breakdown_lines)) if breakdown_lines else ""

        # Why this was escalated.
        reasons = []
        if analysis.fraud_probability >= 0.85:
            reasons.append(
                f"Ensemble model consensus: {fraud_pct:.0f}% fraud probability, triggering "
                f"the automatic-deny threshold. All stochastic sub-models agree."
            )
        elif analysis.fraud_probability >= 0.70:
            reasons.append(
                f"Multiple stochastic models converge on {fraud_pct:.0f}% fraud probability, "
                f"indicating strong and consistent signals of a fraudulent claim."
            )
        if analysis.uncertainty > 0.3:
            reasons.append(
                f"Model uncertainty is elevated ({analysis.uncertainty:.2f}), meaning sub-models "
                f"partially disagree - a human judgment call is required to resolve the ambiguity."
            )
        if ratio > 10:
            reasons.append(
                f"Claim-to-premium ratio is {ratio:.1f}x - critically above normal. "
                f"A ${total_claim:,.0f} claim against this policy's annual premium is a "
                f"primary indicator of inflated or fabricated damage."
            )
        elif ratio > 5:
            reasons.append(
                f"Claim-to-premium ratio of {ratio:.1f}x is substantially above average. "
                f"The claimed amount of ${total_claim:,.0f} appears disproportionate "
                f"to the policy's risk profile."
            )
        if red_flags >= 3:
            reasons.append(
                f"Maximum red-flag score reached ({red_flags} flags). Multiple independent "
                f"risk indicators are simultaneously active on this claim."
            )
        elif red_flags >= 2:
            reasons.append(
                f"{red_flags} concurrent red flags were detected - each independently raises "
                f"suspicion; their combination is statistically rare in legitimate claims."
            )
        if no_witness:
            reasons.append(
                "No independent witness is on record. For a "
                + severity.lower() + " " + incident_type.lower()
                + ", the complete absence of witnesses is statistically uncommon and "
                "removes a key corroborating data point."
            )
        if no_police:
            reasons.append(
                f"No police report was filed despite a {severity.lower()} incident. "
                f"Authorities contacted: {authorities}. This gap in official documentation "
                f"makes independent verification of the event significantly harder."
            )
        if months_cust < 12:
            reasons.append(
                f"The policy is only {months_cust} month(s) old. Fraud is statistically "
                f"more common within the first year of a new policy."
            )
        if hour is not None and (hour < 6 or hour >= 22):
            reasons.append(
                f"Incident reported at {hour:02d}:00 - off-hours incidents have elevated "
                f"fraud rates due to reduced independent verification opportunities."
            )
        if not reasons:
            reasons.append(decision.reasoning)

        # Investigation steps (tailored to this claim).
        steps = []
        if incident_type.lower() in ('single vehicle collision', 'vehicle theft'):
            steps.append(
                f"For a {incident_type.lower()}, independently verify the location, "
                f"road conditions, and sequence of events - these are the most commonly "
                f"fabricated details in single-party claims."
            )
        else:
            steps.append(
                "Obtain statements and contact details from all involved parties and "
                "cross-reference for consistency."
            )
        if vehicle_desc:
            steps.append(
                f"Commission a certified independent appraisal of the {vehicle_desc} "
                f"damage before authorising any repair or replacement payment."
            )
        if ratio > 5:
            steps.append(
                f"Audit all cost components (injury ${injury_claim:,.0f}, property "
                f"${property_claim:,.0f}, vehicle ${vehicle_claim:,.0f}) against "
                f"market-rate schedules and official provider invoices."
            )
        steps.append(
            "Run a full claim history check on the claimant and any co-claimants "
            "for prior losses, frequency patterns, or linked addresses/parties."
        )
        if no_witness:
            steps.append(
                "Conduct an active witness canvass: request CCTV footage from nearby "
                "businesses/traffic systems and check for any social media posts "
                "related to the incident date and location."
            )
        if no_police:
            steps.append(
                f"Contact {authorities} to request any incident record; if none exists, "
                f"this absence should be treated as a material discrepancy in the file."
            )
        if bodily_injuries:
            steps.append(
                f"Verify the {bodily_injuries} reported bodily injur{'y' if bodily_injuries == 1 else 'ies'} "
                f"against independent medical examiner records - soft-tissue claims are "
                f"frequently inflated or fabricated."
            )
        if decision.risk_score >= 85:
            steps.append(
                "Run the claimant's identity, vehicle VIN, and policy details through "
                "industry fraud databases (IFB, NICB, or national insurance bureau equivalent)."
            )
        steps.append(
            "Confirm the policy was in force, premiums were current, and no "
            "material changes were made in the 30 days preceding the incident."
        )

        # Anomaly bullet list.
        anomaly_lines = ""
        if anomalies:
            anomaly_lines = "\n\nANOMALIES DETECTED\n" + "\n".join(f"  - {a}" for a in anomalies)

        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))

        return (
            f"FRAUD RISK ALERT - Manual Review Required [{risk_label}]\n\n"
            f"INCIDENT: {incident_summary}\n\n"
            f"RISK OVERVIEW\n"
            f"  Fraud Probability : {fraud_pct:.0f}%\n"
            f"  Risk Score        : {decision.risk_score:.0f} / 100\n"
            f"  Estimated Loss    : ${analysis.expected_fraud_loss:,.0f} if processed incorrectly\n"
            f"  Review SLA        : {decision.sla_hours} hours\n"
            + claim_breakdown
            + f"\n\nWHY THIS CLAIM WAS ESCALATED\n"
            + "\n".join(f"  - {r}" for r in reasons)
            + anomaly_lines
            + f"\n\nRECOMMENDED INVESTIGATION STEPS\n{steps_text}"
        )
    
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
        
        # Use hybrid explainer when available; otherwise fall back to template generator API.
        if hasattr(self.explainer, 'explain'):
            detailed_explanation = self.explainer.explain(explainer_data)
            source = getattr(self.explainer, 'last_source', None)
            if source not in {'llm', 'template'}:
                source = 'llm' if analysis.fraud_probability > self.explainer.llm_threshold else 'template'
        else:
            generated = self.explainer.generate(decision, analysis)
            detailed_explanation = generated.get('detailed', generated.get('summary', ''))
            source = 'template'
        
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
                print(f"Processed {i+1}/{total} (LLM: {llm_count}, Template: {template_count})")
        
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
        anomalies_text = '|'.join(result['anomalies_detected'])
        
        return {
            'claim_id': decision.claim_id,
            'agent_action': decision.selected_action,
            'final_decision': decision.selected_action,
            'agent_confidence': decision.confidence,
            'agent_reasoning': decision.reasoning,
            'risk_score': decision.risk_score,
            'requires_human_review': decision.requires_human_review,
            'sla_hours': decision.sla_hours,
            'investigation_depth': decision.investigation_depth,
            'fraud_probability': analysis.fraud_probability,
            'expected_fraud_loss': analysis.expected_fraud_loss,
            'explanation_summary': explanations['summary'],
            'explanation': explanations['summary'],
            'explanation_source': explanations.get('source', 'unknown'),
            'human_review_note': result.get('human_review_note', ''),
            'anomalies': anomalies_text
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