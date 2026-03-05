"""
Page 2: Claim Explorer
Individual claim lookup by claim ID
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from layer5.core.dashboard_data import load_decisions, load_features
from layer5.streamlit.components.risk_gauge import risk_gauge
from layer5.streamlit.components.claim_card import claim_card, DECISION_COLORS
from layer5.streamlit.components.decision_timeline import decision_timeline

st.set_page_config(page_title="Claim Explorer — StochClaim", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root { --bg:#0a0c0f; --surface:#111418; --border:#1e2530; --accent:#00e5ff; --text:#e2e8f0; }
html, body, [data-testid="stAppViewContainer"] { background-color:var(--bg) !important;
    font-family:'IBM Plex Sans',sans-serif; color:var(--text); }
[data-testid="stSidebar"] { background-color:var(--surface) !important; border-right:1px solid var(--border); }
h1,h2,h3 { font-family:'IBM Plex Mono',monospace; }
[data-testid="stTextInput"] input { background:#111418 !important; color:#e2e8f0 !important;
    border:1px solid #1e2530 !important; font-family:'IBM Plex Mono',monospace !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; margin-bottom:0.25rem;">
◎ CLAIM EXPLORER
</h2>
<div style="color:#4a5568; font-size:0.8rem; letter-spacing:0.15em; margin-bottom:2rem;">
INDIVIDUAL CLAIM LOOKUP & INSPECTION
</div>
""", unsafe_allow_html=True)

df = load_decisions()

if df.empty:
    st.warning("No processed data found. Run the pipeline first.")
    st.stop()

# Search
col_search, col_btn = st.columns([3, 1])
with col_search:
    search_id = st.text_input("", placeholder="Enter claim ID  (e.g. 791425)", label_visibility="collapsed")
with col_btn:
    search_btn = st.button("SEARCH →", use_container_width=True)

# Show browse list if no search
if not search_id:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1.5rem 0 0.75rem 0;'>BROWSE ALL CLAIMS</div>", unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        decision_filter = st.selectbox("Decision", ["All"] + sorted(df["final_decision"].unique().tolist()) if "final_decision" in df.columns else ["All"])
    with fc2:
        review_filter = st.selectbox("Review Status", ["All", "Needs Review", "Cleared"])
    with fc3:
        risk_filter = st.selectbox("Risk Level", ["All", "Low (<30%)", "Medium (30-60%)", "High (>60%)"])

    filtered = df.copy()
    if decision_filter != "All" and "final_decision" in df.columns:
        filtered = filtered[filtered["final_decision"] == decision_filter]
    if review_filter == "Needs Review" and "requires_human_review" in df.columns:
        filtered = filtered[filtered["requires_human_review"] == True]
    elif review_filter == "Cleared" and "requires_human_review" in df.columns:
        filtered = filtered[filtered["requires_human_review"] == False]
    if risk_filter == "Low (<30%)" and "fraud_probability" in df.columns:
        filtered = filtered[filtered["fraud_probability"] < 0.3]
    elif risk_filter == "Medium (30-60%)" and "fraud_probability" in df.columns:
        filtered = filtered[(filtered["fraud_probability"] >= 0.3) & (filtered["fraud_probability"] < 0.6)]
    elif risk_filter == "High (>60%)" and "fraud_probability" in df.columns:
        filtered = filtered[filtered["fraud_probability"] >= 0.6]

    st.markdown(f"<div style='font-size:0.75rem;color:#4a5568;margin-bottom:0.75rem;'>{len(filtered):,} claims</div>", unsafe_allow_html=True)
    for _, row in filtered.head(30).iterrows():
        claim_card(row.to_dict())

else:
    # Find claim
    id_col = "claim_id" if "claim_id" in df.columns else df.columns[0]
    matches = df[df[id_col].astype(str) == str(search_id)]

    if matches.empty:
        st.markdown(f"""
        <div style="background:#111418; border:1px solid #ff2d5544; border-left:3px solid #ff2d55;
                    padding:1.25rem; border-radius:4px;">
            <span style="font-family:'IBM Plex Mono',monospace; color:#ff2d55;">
                ✕ Claim #{search_id} not found
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        claim = matches.iloc[0].to_dict()
        fraud_prob = float(claim.get("fraud_probability", 0))
        decision = str(claim.get("final_decision", "unknown")).lower()
        color = DECISION_COLORS.get(decision, "#4a5568")

        st.markdown(f"""
        <div style="background:#111418; border:1px solid {color}44; border-left:3px solid {color};
                    padding:1rem 1.5rem; border-radius:4px; margin-bottom:1.5rem;">
            <span style="font-family:'IBM Plex Mono',monospace; color:{color}; font-size:1.1rem;">
                Claim #{search_id}
            </span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([1, 2])
        with r1:
            risk_gauge(fraud_prob)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            decision_timeline(claim)

        with r2:
            st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>CLAIM DETAILS</div>", unsafe_allow_html=True)
            display_fields = {k: v for k, v in claim.items()
                              if k not in ["_ingestion_timestamp", "_source_type", "_batch_id"]
                              and not str(k).startswith("_")}
            cols = st.columns(2)
            items = list(display_fields.items())
            for i, (k, v) in enumerate(items):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="padding:0.5rem 0; border-bottom:1px solid #1e2530;">
                        <div style="font-size:0.65rem; color:#4a5568; letter-spacing:0.1em;">{k.upper().replace('_',' ')}</div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#e2e8f0; margin-top:0.15rem;">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Explanation
        if "explanation" in claim and claim["explanation"]:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#111418; border:1px solid #1e2530; padding:1.25rem; border-radius:4px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                            color:#4a5568; letter-spacing:0.15em; margin-bottom:0.5rem;">AI EXPLANATION</div>
                <div style="font-size:0.9rem; color:#e2e8f0; line-height:1.6;">{claim['explanation']}</div>
            </div>
            """, unsafe_allow_html=True)