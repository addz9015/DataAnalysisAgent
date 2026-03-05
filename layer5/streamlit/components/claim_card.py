"""Claim card component"""
import streamlit as st


DECISION_COLORS = {
    "approve":     "#00ff88",
    "fast_track":  "#00e5ff",
    "standard":    "#ff6b35",
    "deep":        "#ff9500",
    "deny":        "#ff2d55",
    "fraud_detected": "#ff2d55",
}

DECISION_ICONS = {
    "approve":     "✓",
    "fast_track":  "⚡",
    "standard":    "◎",
    "deep":        "◉",
    "deny":        "✕",
    "fraud_detected": "⚠",
}


def claim_card(claim: dict) -> None:
    claim_id = claim.get("claim_id", claim.get("index", "N/A"))
    fraud_prob = float(claim.get("fraud_probability", 0))
    decision = str(claim.get("final_decision", "unknown")).lower()
    color = DECISION_COLORS.get(decision, "#4a5568")
    icon = DECISION_ICONS.get(decision, "?")
    pct = fraud_prob * 100
    review_badge = ""
    if claim.get("requires_human_review"):
        review_badge = '<span style="background:#ff6b3522; border:1px solid #ff6b35; color:#ff6b35; font-size:0.65rem; padding:0.1rem 0.4rem; border-radius:2px; letter-spacing:0.1em;">REVIEW</span>'

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
    </div>
    """, unsafe_allow_html=True)