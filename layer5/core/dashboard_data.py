"""
Layer 5 Core - Dashboard Data
Loads and caches processed data from Layers 1-3
"""

import pandas as pd
import json
import requests
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import streamlit as st

from shared.claim_database import (
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_CLEARED,
    REVIEW_STATUS_DENIED,
    REVIEW_STATUS_IN_PROGRESS,
    REVIEW_STATUS_PENDING,
    get_claim_ids_by_review_status,
    get_claim_review_status,
    load_claim_records,
    set_claim_review_status,
    upsert_claim_prediction,
)

API_BASE = "http://127.0.0.1:8000"
API_KEY = "dev-key"
HEADERS = {"X-API-Key": API_KEY}

logger = logging.getLogger("layer5.dashboard_data")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PROCESSED_CLAIMS_PATH = DATA_DIR / "processed_claims.csv"
UNPROCESSED_CLAIMS_PATH = DATA_DIR / "unprocessed_claims.csv"


def _normalize_claim_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _read_claim_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _reviewed_claims_frame(path: Path, approved: bool) -> pd.DataFrame:
    df = _read_claim_file(path)
    if df.empty or "claim_id" not in df.columns:
        return pd.DataFrame()

    result = df.copy()
    result["claim_id"] = result["claim_id"].map(_normalize_claim_id)
    result["requires_human_review"] = False
    result["human_review_decision"] = "yes" if approved else "no"
    result["human_review_bucket"] = "processed" if approved else "unprocessed"
    result["final_decision"] = "approve" if approved else "deny"
    result["agent_action"] = "approve" if approved else "deny"

    return result.drop_duplicates(subset=["claim_id"], keep="last")


def _apply_human_review_overrides(df: pd.DataFrame) -> pd.DataFrame:
    reviewed_frames = [
        _reviewed_claims_frame(PROCESSED_CLAIMS_PATH, approved=True),
        _reviewed_claims_frame(UNPROCESSED_CLAIMS_PATH, approved=False),
    ]
    reviewed_frames = [f for f in reviewed_frames if not f.empty]
    if not reviewed_frames:
        return df

    reviewed_df = pd.concat(reviewed_frames, ignore_index=True)
    reviewed_df = reviewed_df.drop_duplicates(subset=["claim_id"], keep="last")

    if df.empty:
        return reviewed_df

    if "claim_id" not in df.columns:
        return df

    merged = df.copy()
    merged["claim_id"] = merged["claim_id"].map(_normalize_claim_id)
    merged = merged.set_index("claim_id")

    reviewed_df = reviewed_df.set_index("claim_id")
    review_cols = reviewed_df.columns.tolist()
    common_ids = merged.index.intersection(reviewed_df.index)

    if len(common_ids) > 0:
        for col in review_cols:
            if col not in merged.columns:
                merged[col] = pd.NA
            merged.loc[common_ids, col] = reviewed_df.loc[common_ids, col]

    missing_ids = reviewed_df.index.difference(merged.index)
    if len(missing_ids) > 0:
        merged = pd.concat([merged, reviewed_df.loc[missing_ids]], axis=0, sort=False)

    return merged.reset_index()


def _merge_database_claims(df: pd.DataFrame) -> pd.DataFrame:
    db_records = load_claim_records()
    if not db_records:
        return df

    db_df = pd.DataFrame(db_records)
    if db_df.empty or "claim_id" not in db_df.columns:
        return df

    db_df = db_df.copy()
    db_df["claim_id"] = db_df["claim_id"].map(_normalize_claim_id)

    if df.empty:
        return db_df.drop_duplicates(subset=["claim_id"], keep="last")

    if "claim_id" not in df.columns:
        return df

    merged = df.copy()
    merged["claim_id"] = merged["claim_id"].map(_normalize_claim_id)

    combined = pd.concat([merged, db_df], ignore_index=True, sort=False)
    if "_db_updated_at_utc" in combined.columns:
        combined["_db_updated_at_utc"] = combined["_db_updated_at_utc"].fillna("").astype(str)
        combined = combined.sort_values("_db_updated_at_utc")

    return combined.drop_duplicates(subset=["claim_id"], keep="last")


def _apply_review_status_overrides(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    merged = df.copy()
    if "human_review_status" not in merged.columns:
        return merged

    statuses = merged["human_review_status"].fillna("").astype(str).str.strip().str.lower()
    if "requires_human_review" not in merged.columns:
        merged["requires_human_review"] = False

    pending_mask = statuses == REVIEW_STATUS_PENDING
    in_progress_mask = statuses == REVIEW_STATUS_IN_PROGRESS
    approved_mask = statuses == REVIEW_STATUS_APPROVED
    denied_mask = statuses == REVIEW_STATUS_DENIED
    cleared_mask = statuses == REVIEW_STATUS_CLEARED

    merged.loc[pending_mask | in_progress_mask, "requires_human_review"] = True
    merged.loc[approved_mask | denied_mask | cleared_mask, "requires_human_review"] = False

    if approved_mask.any():
        merged.loc[approved_mask, "human_review_decision"] = "yes"
        merged.loc[approved_mask, "human_review_bucket"] = "processed"
        merged.loc[approved_mask, "final_decision"] = "approve"
        merged.loc[approved_mask, "agent_action"] = "approve"

    if denied_mask.any():
        merged.loc[denied_mask, "human_review_decision"] = "no"
        merged.loc[denied_mask, "human_review_bucket"] = "unprocessed"
        merged.loc[denied_mask, "final_decision"] = "deny"
        merged.loc[denied_mask, "agent_action"] = "deny"

    if in_progress_mask.any():
        merged.loc[in_progress_mask, "human_review_decision"] = "in_progress"
        merged.loc[in_progress_mask, "human_review_bucket"] = "in_progress"

    return merged


def _remove_claim(path: Path, claim_id: str) -> None:
    df = _read_claim_file(path)
    if df.empty or "claim_id" not in df.columns:
        return

    normalized = df["claim_id"].map(_normalize_claim_id)
    trimmed = df[normalized != claim_id].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(path, index=False)


def _upsert_claim(path: Path, claim_record: Dict[str, Any]) -> None:
    claim_id = _normalize_claim_id(claim_record.get("claim_id"))
    incoming = pd.DataFrame([{**claim_record, "claim_id": claim_id}])

    existing = _read_claim_file(path)
    if not existing.empty and "claim_id" in existing.columns:
        existing_ids = existing["claim_id"].map(_normalize_claim_id)
        existing = existing[existing_ids != claim_id].copy()

    merged = pd.concat([existing, incoming], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)


def _persist_review_claim_to_db(claim: Dict[str, Any]) -> None:
    claim_id = _normalize_claim_id(claim.get("claim_id"))
    if claim_id in {"", "nan", "None", "null"}:
        return

    prediction_payload = {
        "claim_id": claim_id,
        "fraud_probability": float(claim.get("fraud_probability", 0) or 0),
        "agent_decision": str(
            claim.get("agent_action")
            or claim.get("final_decision")
            or claim.get("agent_decision")
            or "unknown"
        ),
        "confidence": float(claim.get("agent_confidence", claim.get("confidence", 0)) or 0),
        "risk_score": int(claim.get("risk_score", 0) or 0),
        "requires_human_review": bool(claim.get("requires_human_review", True)),
        "sla_hours": int(claim.get("sla_hours", 0) or 0),
        "investigation_depth": claim.get("investigation_depth"),
        "explanation": str(claim.get("explanation") or claim.get("explanation_summary") or ""),
        "explanation_source": str(claim.get("explanation_source") or "unknown"),
        "human_review_note": str(claim.get("human_review_note") or ""),
    }

    upsert_claim_prediction(
        input_claim={**claim, "claim_id": claim_id},
        prediction=prediction_payload,
        source_type="review_queue",
    )


def get_human_reviewed_ids() -> set[str]:
    reviewed_ids: set[str] = set()
    for path in (PROCESSED_CLAIMS_PATH, UNPROCESSED_CLAIMS_PATH):
        df = _read_claim_file(path)
        if "claim_id" not in df.columns:
            continue
        reviewed_ids.update(
            _normalize_claim_id(v)
            for v in df["claim_id"].tolist()
            if str(v).strip() not in {"", "nan", "None", "null"}
        )

    reviewed_ids.update(
        _normalize_claim_id(v)
        for v in get_claim_ids_by_review_status(
            [REVIEW_STATUS_APPROVED, REVIEW_STATUS_DENIED]
        )
    )

    return reviewed_ids


def get_human_review_record(claim_id: Any) -> Dict[str, str]:
    """Return human review decision metadata for a claim if it exists in review files."""
    normalized_claim_id = _normalize_claim_id(claim_id)
    if normalized_claim_id in {"", "nan", "None", "null"}:
        return {}

    db_review = get_claim_review_status(normalized_claim_id)
    if db_review:
        status = str(db_review.get("human_review_status", "")).strip().lower()
        reviewed_at = str(db_review.get("reviewed_at_utc", "")).strip()

        if status == REVIEW_STATUS_APPROVED:
            return {
                "human_review_bucket": "processed",
                "human_review_decision": "yes",
                "human_review_status": REVIEW_STATUS_APPROVED,
                "reviewed_at_utc": reviewed_at,
            }
        if status == REVIEW_STATUS_DENIED:
            return {
                "human_review_bucket": "unprocessed",
                "human_review_decision": "no",
                "human_review_status": REVIEW_STATUS_DENIED,
                "reviewed_at_utc": reviewed_at,
            }
        if status == REVIEW_STATUS_IN_PROGRESS:
            return {
                "human_review_bucket": "in_progress",
                "human_review_decision": "in_progress",
                "human_review_status": REVIEW_STATUS_IN_PROGRESS,
                "reviewed_at_utc": reviewed_at,
            }
        if status == REVIEW_STATUS_PENDING:
            return {
                "human_review_status": REVIEW_STATUS_PENDING,
                "reviewed_at_utc": reviewed_at,
            }
        if status == REVIEW_STATUS_CLEARED:
            return {
                "human_review_status": REVIEW_STATUS_CLEARED,
                "reviewed_at_utc": reviewed_at,
            }

    review_sources = [
        (PROCESSED_CLAIMS_PATH, "processed", "yes"),
        (UNPROCESSED_CLAIMS_PATH, "unprocessed", "no"),
    ]

    for path, bucket, decision in review_sources:
        df = _read_claim_file(path)
        if df.empty or "claim_id" not in df.columns:
            continue

        normalized_ids = df["claim_id"].map(_normalize_claim_id)
        matched = df[normalized_ids == normalized_claim_id]
        if matched.empty:
            continue

        last = matched.iloc[-1]
        reviewed_at = ""
        if "reviewed_at_utc" in matched.columns:
            raw_reviewed_at = str(last.get("reviewed_at_utc", "")).strip()
            if raw_reviewed_at.lower() not in {"", "nan", "none", "null", "<na>"}:
                reviewed_at = raw_reviewed_at

        return {
            "human_review_bucket": bucket,
            "human_review_decision": decision,
            "reviewed_at_utc": reviewed_at,
        }

    return {}


def record_human_review_decision(claim: Dict[str, Any], approved: bool) -> bool:
    claim_id = _normalize_claim_id(claim.get("claim_id"))
    if claim_id in {"", "nan", "None", "null"}:
        return False

    target_status = REVIEW_STATUS_APPROVED if approved else REVIEW_STATUS_DENIED
    return record_human_review_state(claim=claim, status=target_status)


def record_human_review_state(claim: Dict[str, Any], status: str) -> bool:
    claim_id = _normalize_claim_id(claim.get("claim_id"))
    if claim_id in {"", "nan", "None", "null"}:
        return False

    target_status = str(status).strip().lower()
    if target_status not in {
        REVIEW_STATUS_APPROVED,
        REVIEW_STATUS_DENIED,
        REVIEW_STATUS_IN_PROGRESS,
    }:
        return False

    try:
        if not set_claim_review_status(claim_id, target_status):
            _persist_review_claim_to_db(claim)
            set_claim_review_status(claim_id, target_status)
    except Exception:
        logger.warning("Failed to persist review state for claim %s", claim_id, exc_info=True)

    reviewed_claim = dict(claim)
    reviewed_claim["claim_id"] = claim_id
    reviewed_claim["human_review_status"] = target_status
    reviewed_claim["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()

    if target_status == REVIEW_STATUS_APPROVED:
        reviewed_claim["requires_human_review"] = False
        reviewed_claim["human_review_decision"] = "yes"
        reviewed_claim["human_review_bucket"] = "processed"
        reviewed_claim["final_decision"] = "approve"
        reviewed_claim["agent_action"] = "approve"
        _upsert_claim(PROCESSED_CLAIMS_PATH, reviewed_claim)
        _remove_claim(UNPROCESSED_CLAIMS_PATH, claim_id)
    elif target_status == REVIEW_STATUS_DENIED:
        reviewed_claim["requires_human_review"] = False
        reviewed_claim["human_review_decision"] = "no"
        reviewed_claim["human_review_bucket"] = "unprocessed"
        reviewed_claim["final_decision"] = "deny"
        reviewed_claim["agent_action"] = "deny"
        _upsert_claim(UNPROCESSED_CLAIMS_PATH, reviewed_claim)
        _remove_claim(PROCESSED_CLAIMS_PATH, claim_id)
    elif target_status == REVIEW_STATUS_IN_PROGRESS:
        reviewed_claim["requires_human_review"] = True
        reviewed_claim["human_review_decision"] = "in_progress"
        reviewed_claim["human_review_bucket"] = "in_progress"
        _remove_claim(PROCESSED_CLAIMS_PATH, claim_id)
        _remove_claim(UNPROCESSED_CLAIMS_PATH, claim_id)

    # Ensure pages using cached decisions refresh immediately after review actions.
    load_decisions.clear()

    return True


@st.cache_data(ttl=60)
def load_decisions() -> pd.DataFrame:
    """Load Layer 3 agent decisions"""
    path = DATA_DIR / "agent_decisions.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()

    # The monitor page expects Markov/HMM state columns, which live in layer2 output.
    missing_state_cols = not ({"markov_state", "hmm_state"} & set(df.columns))
    if missing_state_cols and "claim_id" in df.columns:
        layer2_path = DATA_DIR / "layer2_results.csv"
        if layer2_path.exists():
            state_cols = {"claim_id", "markov_state", "hmm_state"}
            layer2_df = pd.read_csv(layer2_path, usecols=lambda c: c in state_cols)
            if not layer2_df.empty and "claim_id" in layer2_df.columns:
                layer2_df = layer2_df.drop_duplicates(subset=["claim_id"], keep="last")
                df = df.copy()
                df["claim_id"] = df["claim_id"].astype(str)
                layer2_df["claim_id"] = layer2_df["claim_id"].astype(str)
                merge_cols = [c for c in ["claim_id", "markov_state", "hmm_state"] if c in layer2_df.columns]
                df = df.merge(layer2_df[merge_cols], on="claim_id", how="left")

    df = _merge_database_claims(df)
    df = _apply_human_review_overrides(df)
    df = _apply_review_status_overrides(df)

    return df


@st.cache_data(ttl=60)
def load_features() -> pd.DataFrame:
    """Load Layer 1 processed features"""
    path = DATA_DIR / "processed_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", headers=HEADERS, timeout=2)
        return r.status_code == 200
    except:
        return False


def api_predict(claim: Dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}/predict/", json=claim, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        detail = r.text
        try:
            payload = r.json()
            detail = payload.get("detail", payload)
        except Exception:
            pass
        st.error(f"API returned {r.status_code}: {detail}")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_batch(claims: list) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}/batch/", json={"claims": claims}, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        detail = r.text
        try:
            payload = r.json()
            detail = payload.get("detail", payload)
        except Exception:
            pass
        st.error(f"API returned {r.status_code}: {detail}")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_explain(claim_id: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}/explain/{claim_id}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def get_summary_stats(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    decision_col = None
    if "final_decision" in df.columns:
        decision_col = "final_decision"
    elif "agent_action" in df.columns:
        decision_col = "agent_action"

    stats = {
        "total_claims": len(df),
        "human_review": int(df.get("requires_human_review", pd.Series([False]*len(df))).sum()),
        "avg_fraud_prob": float(df["fraud_probability"].mean()) if "fraud_probability" in df.columns else 0,
        "high_risk": int((df["fraud_probability"] > 0.7).sum()) if "fraud_probability" in df.columns else 0,
    }
    if decision_col:
        stats["decision_dist"] = df[decision_col].value_counts().to_dict()
    if "markov_state" in df.columns:
        stats["markov_dist"] = df["markov_state"].value_counts().to_dict()
    return stats