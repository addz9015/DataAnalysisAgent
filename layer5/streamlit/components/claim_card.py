"""Claim card component"""
import html
import math
import streamlit as st


DECISION_COLORS = {
    "approve":     "#00ff88",
    "fast_track":  "#00e5ff",
    "standard":    "#ff6b35",
    "deep":        "#ff9500",
    "deny":        "#ff2d55",
    "in_progress": "#ffd166",
    "fraud_detected": "#ff2d55",
}

DECISION_ICONS = {
    "approve":     "✓",
    "fast_track":  "⚡",
    "standard":    "◎",
    "deep":        "◉",
    "deny":        "✕",
    "in_progress": "◌",
    "fraud_detected": "⚠",
}


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True

    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _clean_text(value, default="") -> str:
    return default if _is_missing(value) else str(value).strip()


def _coerce_float(value, default=0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(val):
        return default
    return val


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def claim_card(claim: dict) -> None:
    claim_id = _clean_text(claim.get("claim_id", claim.get("index", "N/A")), "N/A")
    fraud_prob = _coerce_float(claim.get("fraud_probability", 0.0), 0.0)

    decision = "unknown"
    for candidate in (
        claim.get("final_decision"),
        claim.get("agent_action"),
        claim.get("agent_decision"),
    ):
        cleaned = _clean_text(candidate, "")
        if cleaned:
            decision = cleaned.lower()
            break

    color = DECISION_COLORS.get(decision, "#4a5568")
    icon = DECISION_ICONS.get(decision, "?")
    pct = max(0, min(fraud_prob * 100, 100))

    requires_review = _as_bool(claim.get("requires_human_review"))
    review_status = _clean_text(claim.get("human_review_status"), "").lower()
    if review_status == "in_progress":
        decision = "in_progress"
        color = DECISION_COLORS["in_progress"]
        icon = DECISION_ICONS["in_progress"]

    review_badge = ""
    if review_status == "in_progress":
        review_badge = '<span style="background:#ffd16622; border:1px solid #ffd166; color:#ffd166; font-size:0.65rem; padding:0.1rem 0.4rem; border-radius:2px; letter-spacing:0.1em;">IN-PROGRESS</span>'
    elif requires_review:
        review_badge = '<span style="background:#ff6b3522; border:1px solid #ff6b35; color:#ff6b35; font-size:0.65rem; padding:0.1rem 0.4rem; border-radius:2px; letter-spacing:0.1em;">REVIEW</span>'

    review_note = _clean_text(claim.get("human_review_note"), "")
    if not review_note and requires_review:
        anomalies = _clean_text(claim.get("anomalies"), "")
        anomalies = "; ".join([a.strip() for a in anomalies.split("|") if a.strip()])
        rationale = (
            _clean_text(claim.get("agent_reasoning"), "")
            or _clean_text(claim.get("explanation_summary"), "")
            or _clean_text(claim.get("explanation"), "")
        )
        if anomalies and rationale:
            review_note = f"Suspicious signals: {anomalies}. Agent rationale: {rationale}"
        elif anomalies:
            review_note = f"Suspicious signals: {anomalies}."
        elif rationale:
            review_note = f"Agent rationale: {rationale}"

    review_note_html = ""
    if review_note and requires_review:
        safe_note = html.escape(review_note).replace("\n", "<br>")
        review_note_html = (
            '<div style="margin-top:0.6rem;padding:0.55rem 0.65rem;background:#2a1418;'
            'border:1px solid #ff2d5544;border-left:2px solid #ff6b35;'
            'font-size:0.74rem;color:#ffd7d7;line-height:1.45;border-radius:3px;">'
            + safe_note
            + "</div>"
        )

    st.markdown(f"""
    <div style="background:#111418; border:1px solid #1e2530; border-left:3px solid {color};
                padding:1rem 1.25rem; border-radius:4px; margin-bottom:0.5rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#e2e8f0;">
                    #{claim_id}
                </span>
                &nbsp;&nbsp;{review_badge}
            </div>
            <div style="display:flex; align-items:center; gap:1rem;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:{color};">
                    {icon} {decision.upper().replace('_',' ')}
                </span>
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem;
                             color:{color}; font-weight:600;">{pct:.1f}%</span>
            </div>
        </div>
        <div style="margin-top:0.5rem; background:#0a0c0f; border-radius:2px; height:3px;">
            <div style="background:{color}; width:{pct}%; height:100%; border-radius:2px;
                        box-shadow:0 0 6px {color}66;"></div>
        </div>
        {review_note_html}
    </div>
    """, unsafe_allow_html=True)