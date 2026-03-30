"""SQLite persistence for manual and batch claim entries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging
from contextlib import contextmanager


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
DB_PATH = DATA_DIR / "claims.db"

REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_IN_PROGRESS = "in_progress"
REVIEW_STATUS_APPROVED = "approved"
REVIEW_STATUS_DENIED = "denied"
REVIEW_STATUS_CLEARED = "cleared"

VALID_REVIEW_STATUSES = {
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_IN_PROGRESS,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_DENIED,
    REVIEW_STATUS_CLEARED,
}

logger = logging.getLogger("shared.claim_database")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_claim_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _ensure_db_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS claim_records (
            claim_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            input_payload TEXT NOT NULL,
            fraud_probability REAL,
            agent_decision TEXT,
            confidence REAL,
            risk_score INTEGER,
            requires_human_review INTEGER NOT NULL DEFAULT 0,
            sla_hours INTEGER,
            investigation_depth INTEGER,
            explanation TEXT,
            explanation_source TEXT,
            human_review_note TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            review_updated_at_utc TEXT,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_claim_records_review_status
        ON claim_records (review_status)
        """
    )


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.DatabaseError:
        logger.warning("Could not apply one or more SQLite pragmas", exc_info=True)
    _ensure_db_schema(conn)
    return conn


@contextmanager
def _connection() -> sqlite3.Connection:
    """Yield a SQLite connection that is always closed."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _default_review_status(requires_human_review: bool) -> str:
    return REVIEW_STATUS_PENDING if requires_human_review else REVIEW_STATUS_CLEARED


def _serialize_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)


def _prediction_tuple(
    input_claim: Dict[str, Any],
    prediction: Dict[str, Any],
    source_type: str,
) -> Tuple[Any, ...]:
    claim_id = normalize_claim_id(prediction.get("claim_id") or input_claim.get("claim_id"))
    if not claim_id:
        raise ValueError("claim_id is required for persistence")

    now = _utc_now_iso()
    requires_human_review = bool(prediction.get("requires_human_review", False))
    review_status = _default_review_status(requires_human_review)

    return (
        claim_id,
        source_type,
        _serialize_payload({**input_claim, "claim_id": claim_id}),
        float(prediction.get("fraud_probability", 0.0)),
        str(prediction.get("agent_decision") or prediction.get("agent_action") or "unknown"),
        float(prediction.get("confidence", 0.0)),
        int(prediction.get("risk_score", 0) or 0),
        1 if requires_human_review else 0,
        int(prediction.get("sla_hours", 0) or 0),
        int(prediction.get("investigation_depth", 0) or 0) if prediction.get("investigation_depth") is not None else None,
        str(prediction.get("explanation", "")),
        str(prediction.get("explanation_source", "unknown")),
        str(prediction.get("human_review_note", "") or ""),
        review_status,
        now,
        now,
    )


def _prediction_upsert_sql() -> str:
    return (
        """
        INSERT INTO claim_records (
            claim_id,
            source_type,
            input_payload,
            fraud_probability,
            agent_decision,
            confidence,
            risk_score,
            requires_human_review,
            sla_hours,
            investigation_depth,
            explanation,
            explanation_source,
            human_review_note,
            review_status,
            created_at_utc,
            updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(claim_id) DO UPDATE SET
            source_type = excluded.source_type,
            input_payload = excluded.input_payload,
            fraud_probability = excluded.fraud_probability,
            agent_decision = excluded.agent_decision,
            confidence = excluded.confidence,
            risk_score = excluded.risk_score,
            requires_human_review = CASE
                WHEN claim_records.review_status IN ('approved', 'denied', 'in_progress')
                     AND excluded.review_status IN ('pending', 'cleared')
                THEN claim_records.requires_human_review
                ELSE excluded.requires_human_review
            END,
            sla_hours = excluded.sla_hours,
            investigation_depth = excluded.investigation_depth,
            explanation = excluded.explanation,
            explanation_source = excluded.explanation_source,
            human_review_note = excluded.human_review_note,
            review_status = CASE
                WHEN claim_records.review_status IN ('approved', 'denied', 'in_progress')
                     AND excluded.review_status IN ('pending', 'cleared')
                THEN claim_records.review_status
                ELSE excluded.review_status
            END,
            updated_at_utc = excluded.updated_at_utc,
            review_updated_at_utc = CASE
                WHEN claim_records.review_status != (
                    CASE
                        WHEN claim_records.review_status IN ('approved', 'denied', 'in_progress')
                             AND excluded.review_status IN ('pending', 'cleared')
                        THEN claim_records.review_status
                        ELSE excluded.review_status
                    END
                )
                THEN excluded.updated_at_utc
                ELSE claim_records.review_updated_at_utc
            END
        """
    )


def upsert_claim_prediction(
    input_claim: Dict[str, Any],
    prediction: Dict[str, Any],
    source_type: str,
) -> None:
    row = _prediction_tuple(input_claim=input_claim, prediction=prediction, source_type=source_type)
    with _connection() as conn:
        conn.execute(_prediction_upsert_sql(), row)


def upsert_claim_predictions(
    entries: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    source_type: str,
) -> None:
    if not entries:
        return

    rows = [
        _prediction_tuple(input_claim=input_claim, prediction=prediction, source_type=source_type)
        for input_claim, prediction in entries
    ]
    with _connection() as conn:
        conn.executemany(_prediction_upsert_sql(), rows)


def load_claim_records() -> List[Dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT
                claim_id,
                source_type,
                input_payload,
                fraud_probability,
                agent_decision,
                confidence,
                risk_score,
                requires_human_review,
                sla_hours,
                investigation_depth,
                explanation,
                explanation_source,
                human_review_note,
                review_status,
                review_updated_at_utc,
                created_at_utc,
                updated_at_utc
            FROM claim_records
            ORDER BY updated_at_utc ASC
            """
        ).fetchall()

    records: List[Dict[str, Any]] = []
    for row in rows:
        payload_data: Dict[str, Any] = {}
        raw_payload = row["input_payload"]
        if raw_payload:
            try:
                parsed = json.loads(raw_payload)
                if isinstance(parsed, dict):
                    payload_data = parsed
            except json.JSONDecodeError:
                payload_data = {}

        record: Dict[str, Any] = {
            "claim_id": row["claim_id"],
            "source_type": row["source_type"],
            "fraud_probability": row["fraud_probability"],
            "agent_action": row["agent_decision"],
            "final_decision": row["agent_decision"],
            "agent_decision": row["agent_decision"],
            "agent_confidence": row["confidence"],
            "confidence": row["confidence"],
            "risk_score": row["risk_score"],
            "requires_human_review": bool(row["requires_human_review"]),
            "sla_hours": row["sla_hours"],
            "investigation_depth": row["investigation_depth"],
            "explanation": row["explanation"],
            "explanation_summary": row["explanation"],
            "explanation_source": row["explanation_source"],
            "human_review_note": row["human_review_note"],
            "human_review_status": row["review_status"],
            "reviewed_at_utc": row["review_updated_at_utc"],
            "_db_created_at_utc": row["created_at_utc"],
            "_db_updated_at_utc": row["updated_at_utc"],
        }

        for key, value in payload_data.items():
            if key not in record:
                record[key] = value

        records.append(record)

    return records


def get_claim_review_status(claim_id: Any) -> Dict[str, str]:
    normalized = normalize_claim_id(claim_id)
    if not normalized:
        return {}

    with _connection() as conn:
        row = conn.execute(
            """
            SELECT review_status, review_updated_at_utc
            FROM claim_records
            WHERE claim_id = ?
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        return {}

    return {
        "human_review_status": str(row["review_status"] or ""),
        "reviewed_at_utc": str(row["review_updated_at_utc"] or ""),
    }


def set_claim_review_status(claim_id: Any, review_status: str) -> bool:
    normalized = normalize_claim_id(claim_id)
    status = str(review_status).strip().lower()
    if not normalized or status not in VALID_REVIEW_STATUSES:
        return False

    now = _utc_now_iso()
    requires_human_review = 1 if status in {REVIEW_STATUS_PENDING, REVIEW_STATUS_IN_PROGRESS} else 0

    with _connection() as conn:
        result = conn.execute(
            """
            UPDATE claim_records
            SET
                review_status = ?,
                requires_human_review = ?,
                review_updated_at_utc = ?,
                updated_at_utc = ?
            WHERE claim_id = ?
            """,
            (status, requires_human_review, now, now, normalized),
        )
        return result.rowcount > 0


def get_claim_ids_by_review_status(statuses: Iterable[str]) -> set[str]:
    normalized_statuses = [s.strip().lower() for s in statuses if s and s.strip().lower() in VALID_REVIEW_STATUSES]
    if not normalized_statuses:
        return set()

    placeholders = ",".join("?" for _ in normalized_statuses)
    query = (
        "SELECT claim_id FROM claim_records "
        f"WHERE review_status IN ({placeholders})"
    )

    with _connection() as conn:
        rows = conn.execute(query, normalized_statuses).fetchall()

    return {str(row["claim_id"]) for row in rows}
