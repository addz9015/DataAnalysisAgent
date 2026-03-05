"""Risk gauge component"""
import streamlit as st


def risk_gauge(fraud_prob: float, label: str = "Fraud Probability") -> None:
    pct = fraud_prob * 100
    if pct < 30:
        color = "#00ff88"
        risk_label = "LOW RISK"
    elif pct < 60:
        color = "#ff6b35"
        risk_label = "MEDIUM RISK"
    else:
        color = "#ff2d55"
        risk_label = "HIGH RISK"

    st.markdown(f"""
    <div style="background:#111418; border:1px solid #1e2530; padding:1.5rem; border-radius:4px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                    color:#4a5568; letter-spacing:0.15em; margin-bottom:0.75rem;">
            {label.upper()}
        </div>
        <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.75rem;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
                         color:{color}; font-weight:600;">{pct:.1f}%</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                         color:{color}; letter-spacing:0.15em;">{risk_label}</span>
        </div>
        <div style="background:#0a0c0f; border-radius:2px; height:6px; overflow:hidden;">
            <div style="background:{color}; width:{pct}%; height:100%;
                        transition:width 0.5s ease; border-radius:2px;
                        box-shadow:0 0 8px {color}88;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)