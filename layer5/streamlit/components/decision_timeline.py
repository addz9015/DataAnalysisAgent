"""Decision timeline component."""
import html
import math

import streamlit as st


STAGES = [
    ("L1", "Data Intake", "Ingestion & validation"),
    ("L2", "Ensemble", "Stochastic model scoring"),
    ("L3", "Agent", "Decision & routing"),
    ("L4", "API", "Result served"),
]


def _clean_text(value, default="") -> str:
    """Normalize text-like values and hide null-like placeholders."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def decision_timeline(claim: dict) -> None:
    try:
        fraud_prob = float(claim.get("fraud_probability", 0))
    except (TypeError, ValueError):
        fraud_prob = 0.0
    if math.isnan(fraud_prob):
        fraud_prob = 0.0

    decision = _clean_text(
        claim.get("final_decision")
        or claim.get("agent_action")
        or claim.get("agent_decision"),
        "unknown",
    ).lower()
    review_status = _clean_text(claim.get("human_review_status"), "").lower()
    if review_status == "in_progress":
        decision = "in_progress"

    markov = _clean_text(claim.get("markov_state") or claim.get("hmm_state"), "N/A")

    if decision == "in_progress":
        active_color = "#ffd166"
    elif fraud_prob < 0.3:
        active_color = "#00ff88"
    elif fraud_prob < 0.6:
        active_color = "#ff6b35"
    else:
        active_color = "#ff2d55"

    stage_chunks = []
    for i, (code, name, desc) in enumerate(STAGES):
        c = active_color if i < 3 else "#4a5568"
        connector = (
            f'<div style="width:32px;min-width:32px;height:1px;background:{c};opacity:0.85;"></div>'
            if i < len(STAGES) - 1 else ""
        )
        circle = (
            f'<div style="width:36px;height:36px;border-radius:50%;border:2px solid {c};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:{c};'
            f'box-shadow:0 0 8px {c}44;">{html.escape(code)}</div>'
        )
        label = f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:{c};">{html.escape(name)}</div>'
        sub = f'<div style="font-size:0.6rem;color:#4a5568;text-align:center;max-width:90px;">{html.escape(desc)}</div>'
        node = f'<div style="display:flex;flex-direction:column;align-items:center;gap:0.25rem;min-width:86px;">{circle}{label}{sub}</div>'
        stage_chunks.append(
            f'<div style="display:flex;align-items:center;gap:0.35rem;flex:0 0 auto;">{node}{connector}</div>'
        )

    stages_html = "".join(stage_chunks)
    decision_upper = html.escape(decision.upper().replace("_", " "))
    markov_safe = html.escape(markov)

    timeline_html = (
        '<div style="background:#111418;border:1px solid #1e2530;padding:1.5rem;border-radius:4px;">'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
        'color:#4a5568;letter-spacing:0.15em;margin-bottom:1.25rem;">PIPELINE TRACE</div>'
        '<div style="display:flex;align-items:center;justify-content:flex-start;'
        f'flex-wrap:nowrap;overflow-x:auto;gap:0.15rem;padding-bottom:0.2rem;">{stages_html}</div>'
        '<div style="margin-top:1.25rem;padding-top:1rem;border-top:1px solid #1e2530;'
        'display:flex;gap:2rem;flex-wrap:wrap;">'
        '<div><div style="font-size:0.65rem;color:#4a5568;letter-spacing:0.1em;">MARKOV STATE</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;color:#00e5ff;margin-top:0.2rem;">{markov_safe}</div></div>'
        '<div><div style="font-size:0.65rem;color:#4a5568;letter-spacing:0.1em;">FINAL DECISION</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;color:{active_color};margin-top:0.2rem;">{decision_upper}</div></div>'
        '</div>'
        '</div>'
    )
    st.markdown(timeline_html, unsafe_allow_html=True)