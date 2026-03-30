import pytest

from layer3.core.action_selector import ActionSelector
from layer3.core.reasoning_engine import SituationAnalysis


def _options(primary_action: str):
    return [
        {
            "action": primary_action,
            "expected_value": 100.0,
            "fraud_prevention_potential": 0.5,
            "cost": 100.0,
        },
        {
            "action": "standard",
            "expected_value": 80.0,
            "fraud_prevention_potential": 0.3,
            "cost": 200.0,
        },
        {
            "action": "deny",
            "expected_value": 60.0,
            "fraud_prevention_potential": 0.7,
            "cost": 100.0,
        },
    ]


def test_low_risk_documented_claim_uses_sanity_override():
    selector = ActionSelector()
    analysis = SituationAnalysis(
        claim_id="SANITY_NON_FRAUD_001",
        fraud_probability=0.90,
        uncertainty=0.05,
        expected_fraud_loss=500.0,
        investigation_cost={},
        time_pressure="low",
        risk_tolerance="balanced",
        key_evidence={
            "red_flags": 0,
            "claim_premium_ratio": 1.8,
            "severity": "Minor Damage",
            "witness_present": "Yes",
            "police_report": "Yes",
        },
    )

    decision = selector.decide(analysis, _options("deny"), anomalies=[])

    assert decision.sanity_override is True
    assert decision.selected_action == "approve"
    assert decision.requires_human_review is False
    assert decision.risk_score <= 20


def test_high_risk_claim_does_not_use_sanity_override():
    selector = ActionSelector()
    analysis = SituationAnalysis(
        claim_id="SANITY_FRAUD_001",
        fraud_probability=0.90,
        uncertainty=0.10,
        expected_fraud_loss=15000.0,
        investigation_cost={},
        time_pressure="high",
        risk_tolerance="balanced",
        key_evidence={
            "red_flags": 4,
            "claim_premium_ratio": 89.2,
            "severity": "Total Loss",
            "witness_present": "No",
            "police_report": "No",
        },
    )

    decision = selector.decide(
        analysis,
        _options("deep"),
        anomalies=["Maximum red flag count - highly suspicious pattern"],
    )

    assert decision.sanity_override is False
    assert decision.selected_action == "deep"
    assert decision.requires_human_review is True
