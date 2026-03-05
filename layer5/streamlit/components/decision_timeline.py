"""Decision timeline component"""
import streamlit as st


STAGES = [
    ("L1", "Data Intake", "Ingestion & validation"),
    ("L2", "Ensemble", "Stochastic model scoring"),
    ("L3", "Agent", "Decision & routing"),
    ("L4", "API", "Result served"),
]


def decision_timeline(claim: dict) -> None:
    fraud_prob = float(claim.get("fraud_probability", 0))
    decision = str(claim.get("final_decision", "unknown")).lower()
    markov = claim.get("markov_state", "Unknown")

    if fraud_prob < 0.3:
        active_color = "#00ff88"
    elif fraud_prob < 0.6:
        active_color = "#ff6b35"
    else:
        active_color = "#ff2d55"

    stages_html = ""
    for i, (code, name, desc) in enumerate(STAGES):
        c = active_color if i < 3 else "#4a5568"
        connector = f'<div style="width:40px; height:1px; background:{c}; margin:0 0.25rem;"></div>' if i < len(STAGES)-1 else ""
        stages_html += f"""
        <div style="display:flex; flex-direction:column; align-items:center; gap:0.25rem;">
            <div style="width:36px; height:36px; border-radius:50%; border:2px solid {c};
                        display:flex; align-items:center; justify-content:center;
                        font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:{c};
                        box-shadow:0 0 8px {c}44;">
                {code}
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:{c};">{name}</div>
            <div style="font-size:0.6rem; color:#4a5568; text-align:center; max-width:80px;">{desc}</div>
        </div>
        {connector if i < len(STAGES)-1 else ""}
        """

    st.markdown(f"""
    <div style="background:#111418; border:1px solid #1e2530; padding:1.5rem; border-radius:4px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                    color:#4a5568; letter-spacing:0.15em; margin-bottom:1.25rem;">
            PIPELINE TRACE
        </div>
        <div style="display:flex; align-items:flex-start; justify-content:center; flex-wrap:wrap; gap:0.25rem;">
            {stages_html}
        </div>
        <div style="margin-top:1.25rem; padding-top:1rem; border-top:1px solid #1e2530;
                    display:flex; gap:2rem;">
            <div>
                <div style="font-size:0.65rem; color:#4a5568; letter-spacing:0.1em;">MARKOV STATE</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                             color:#00e5ff; margin-top:0.2rem;">{markov}</div>
            </div>
            <div>
                <div style="font-size:0.65rem; color:#4a5568; letter-spacing:0.1em;">FINAL DECISION</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                             color:{active_color}; margin-top:0.2rem;">{decision.upper().replace('_',' ')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)