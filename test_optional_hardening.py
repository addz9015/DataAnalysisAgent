import pytest
from fastapi.testclient import TestClient

from layer4.main import app
import layer4.routers.predict as predict_router
from shared import claim_database as claim_db


class _StubPipeline:
    def __init__(self):
        self.seen_df = None

    def process(self, df, export_csv=False):
        self.seen_df = df.copy()
        return df.copy(), None


class _StubEnsemble:
    def predict(self, processed):
        return processed.copy()


class _StubDecision:
    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.selected_action = "approve"
        self.confidence = 0.8
        self.risk_score = 20
        self.requires_human_review = False
        self.sla_hours = 24
        self.investigation_depth = None


class _StubAnalysis:
    def __init__(self):
        self.fraud_probability = 0.2


class _StubAgent:
    def process_claim(self, row):
        claim_id = str(row.get("claim_id", "ID_TEST"))
        return {
            "decision": _StubDecision(claim_id),
            "analysis": _StubAnalysis(),
            "explanations": {"summary": "ok", "source": "template"},
            "human_review_note": "",
        }


@pytest.fixture
def client():
    return TestClient(app)


def test_empty_batch_returns_422(client):
    response = client.post(
        "/batch/",
        headers={"X-API-Key": "dev-key"},
        json={"claims": []},
    )
    assert response.status_code == 422


def test_predict_infers_witness_count_from_presence(client, monkeypatch):
    pipeline = _StubPipeline()

    monkeypatch.setattr(predict_router, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(predict_router, "get_ensemble", lambda: _StubEnsemble())
    monkeypatch.setattr(predict_router, "get_agent", lambda: _StubAgent())
    monkeypatch.setattr(
        predict_router,
        "upsert_claim_prediction",
        lambda *args, **kwargs: None,
    )

    payload = {
        "months_as_customer": 24,
        "age": 35,
        "policy_annual_premium": 1200,
        "incident_severity": "Minor Damage",
        "total_claim_amount": 800,
        "injury_claim": 100,
        "property_claim": 200,
        "vehicle_claim": 500,
        "incident_type": "Parked Car",
        "collision_type": "No Collision",
        "authorities_contacted": "Police",
        "witness_present": "Yes",
        "police_report_available": "Yes",
    }

    response = client.post(
        "/predict/",
        headers={"X-API-Key": "dev-key"},
        json=payload,
    )

    assert response.status_code == 200, response.text
    assert pipeline.seen_df is not None
    assert int(pipeline.seen_df.iloc[0]["witnesses"]) == 1


def test_review_status_preserved_across_prediction_upserts(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    db_path = data_dir / "claims.db"

    monkeypatch.setattr(claim_db, "DATA_DIR", data_dir)
    monkeypatch.setattr(claim_db, "DB_PATH", db_path)

    claim_id = "PRESERVE_REVIEW_STATUS_1"
    claim = {"claim_id": claim_id, "months_as_customer": 12}

    first_prediction = {
        "claim_id": claim_id,
        "fraud_probability": 0.90,
        "agent_decision": "deny",
        "confidence": 0.80,
        "risk_score": 90,
        "requires_human_review": True,
        "sla_hours": 72,
        "investigation_depth": None,
        "explanation": "first",
        "explanation_source": "template",
        "human_review_note": "review",
    }

    claim_db.upsert_claim_prediction(claim, first_prediction, source_type="single_entry")
    assert claim_db.set_claim_review_status(claim_id, claim_db.REVIEW_STATUS_APPROVED)

    before = claim_db.get_claim_review_status(claim_id)

    second_prediction = {
        "claim_id": claim_id,
        "fraud_probability": 0.95,
        "agent_decision": "deny",
        "confidence": 0.75,
        "risk_score": 95,
        "requires_human_review": True,
        "sla_hours": 72,
        "investigation_depth": None,
        "explanation": "second",
        "explanation_source": "template",
        "human_review_note": "review2",
    }

    claim_db.upsert_claim_prediction(claim, second_prediction, source_type="single_entry")
    after = claim_db.get_claim_review_status(claim_id)

    assert before["human_review_status"] == claim_db.REVIEW_STATUS_APPROVED
    assert after["human_review_status"] == claim_db.REVIEW_STATUS_APPROVED

    records = claim_db.load_claim_records()
    record = next(r for r in records if r["claim_id"] == claim_id)
    assert record["requires_human_review"] is False
