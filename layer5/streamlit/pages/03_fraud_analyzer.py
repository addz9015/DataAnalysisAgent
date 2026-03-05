"""
Page 3: Fraud Analyzer
Batch upload new claims and run through pipeline
"""

import streamlit as st
import pandas as pd
import json
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from layer5.core.dashboard_data import api_predict, api_batch, api_health
from layer5.streamlit.components.risk_gauge import risk_gauge
from layer5.streamlit.components.claim_card import claim_card

st.set_page_config(page_title="Fraud Analyzer — StochClaim", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root { --bg:#0a0c0f; --surface:#111418; --border:#1e2530; --accent:#00e5ff; --text:#e2e8f0; }
html, body, [data-testid="stAppViewContainer"] { background-color:var(--bg) !important;
    font-family:'IBM Plex Sans',sans-serif; color:var(--text); }
[data-testid="stSidebar"] { background-color:var(--surface) !important; border-right:1px solid var(--border); }
h1,h2,h3 { font-family:'IBM Plex Mono',monospace; }
[data-testid="stFileUploader"] { background:#111418 !important; border:1px dashed #1e2530 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; margin-bottom:0.25rem;">
◉ FRAUD ANALYZER
</h2>
<div style="color:#4a5568; font-size:0.8rem; letter-spacing:0.15em; margin-bottom:2rem;">
BATCH UPLOAD & REAL-TIME ANALYSIS
</div>
""", unsafe_allow_html=True)

# API status
is_online = api_health()
status_color = "#00ff88" if is_online else "#ff2d55"
status_text = "API ONLINE" if is_online else "API OFFLINE"
st.markdown(f"""
<div style="display:inline-flex; align-items:center; gap:0.5rem; background:#111418;
            border:1px solid #1e2530; padding:0.4rem 0.75rem; border-radius:2px; margin-bottom:1.5rem;">
    <div style="width:6px; height:6px; border-radius:50%; background:{status_color};
                box-shadow:0 0 6px {status_color};"></div>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                 color:{status_color}; letter-spacing:0.15em;">{status_text}</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 FILE UPLOAD", "✏️ MANUAL ENTRY"])

with tab1:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>UPLOAD CLAIMS FILE</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop file here",
        type=["csv", "json", "xlsx", "parquet"],
        label_visibility="collapsed"
    )

    if uploaded:
        try:
            ext = uploaded.name.split(".")[-1].lower()
            if ext == "csv":
                df_upload = pd.read_csv(uploaded)
            elif ext == "json":
                df_upload = pd.read_json(uploaded)
            elif ext == "xlsx":
                df_upload = pd.read_excel(uploaded)
            elif ext == "parquet":
                df_upload = pd.read_parquet(uploaded)

            st.markdown(f"""
            <div style="background:#111418; border:1px solid #00ff8844; border-left:3px solid #00ff88;
                        padding:0.75rem 1rem; border-radius:4px; margin:0.75rem 0;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#00ff88;">
                    ✓ Loaded {len(df_upload):,} claims from {uploaded.name}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(df_upload.head(5), use_container_width=True, hide_index=True)

            if st.button("▶ RUN THROUGH PIPELINE", use_container_width=True):
                if not is_online:
                    st.error("API is offline. Start Layer 4 first: python run_layer4.py")
                else:
                    claims = df_upload.to_dict(orient="records")
                    with st.spinner("Processing through pipeline..."):
                        result = api_batch(claims)
                    if result:
                        st.success(f"✓ Processed {len(result.get('results', []))} claims")
                        results_df = pd.DataFrame(result.get("results", []))
                        st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1rem 0 0.5rem 0;'>RESULTS</div>", unsafe_allow_html=True)
                        for _, row in results_df.iterrows():
                            claim_card(row.to_dict())
                        csv_out = results_df.to_csv(index=False)
                        st.download_button("⬇ DOWNLOAD RESULTS CSV", csv_out,
                                           file_name="fraud_analysis_results.csv", mime="text/csv")
                    else:
                        st.error("Pipeline returned no results. Check Layer 4 logs.")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

with tab2:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:1rem;'>ENTER SINGLE CLAIM</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        months_as_customer = st.number_input("Months as Customer", min_value=0, value=24)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        policy_annual_premium = st.number_input("Annual Premium ($)", min_value=0.0, value=1200.0)
    with c2:
        total_claim_amount = st.number_input("Total Claim ($)", min_value=0.0, value=8000.0)
        injury_claim = st.number_input("Injury Claim ($)", min_value=0.0, value=2000.0)
        property_claim = st.number_input("Property Claim ($)", min_value=0.0, value=3000.0)
    with c3:
        vehicle_claim = st.number_input("Vehicle Claim ($)", min_value=0.0, value=3000.0)
        incident_severity = st.selectbox("Incident Severity",
            ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
        witness_present = st.selectbox("Witness Present", ["YES", "NO"])
        police_report = st.selectbox("Police Report", ["YES", "NO"])

    incident_type = st.selectbox("Incident Type",
        ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"])
    collision_type = st.selectbox("Collision Type",
        ["Front Collision", "Rear Collision", "Side Collision", "?"])
    authorities = st.selectbox("Authorities Contacted",
        ["Police", "Fire", "Ambulance", "None", "Other"])

    if st.button("▶ ANALYZE CLAIM", use_container_width=True):
        if not is_online:
            st.error("API is offline. Start Layer 4 first: python run_layer4.py")
        else:
            claim = {
                "months_as_customer": months_as_customer,
                "age": age,
                "policy_annual_premium": policy_annual_premium,
                "total_claim_amount": total_claim_amount,
                "injury_claim": injury_claim,
                "property_claim": property_claim,
                "vehicle_claim": vehicle_claim,
                "incident_severity": incident_severity,
                "incident_type": incident_type,
                "collision_type": collision_type,
                "authorities_contacted": authorities,
                "witness_present": witness_present,
                "police_report_available": police_report,
            }
            with st.spinner("Analyzing..."):
                result = api_predict(claim)
            if result:
                fraud_prob = float(result.get("fraud_probability", 0))
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                r1, r2 = st.columns([1, 2])
                with r1:
                    risk_gauge(fraud_prob)
                with r2:
                    claim_card(result)
                    if result.get("explanation"):
                        st.markdown(f"""
                        <div style="background:#111418; border:1px solid #1e2530;
                                    padding:1rem; border-radius:4px; margin-top:0.75rem;">
                            <div style="font-size:0.65rem; color:#4a5568; letter-spacing:0.1em;
                                        margin-bottom:0.5rem;">EXPLANATION</div>
                            <div style="font-size:0.85rem; color:#e2e8f0; line-height:1.6;">
                                {result['explanation']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Analysis failed. Check Layer 4 API logs.")