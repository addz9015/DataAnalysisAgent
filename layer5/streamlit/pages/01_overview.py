"""
Page 1: Overview Dashboard
Fraud stats, claim distributions, key metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from layer5.core.dashboard_data import load_decisions, get_summary_stats
from layer5.streamlit.components.navigation import render_navigation

st.set_page_config(page_title="Overview — StochClaim", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root { --bg:#0a0c0f; --surface:#111418; --border:#1e2530; --accent:#00e5ff;
        --warn:#ff6b35; --danger:#ff2d55; --ok:#00ff88; --muted:#4a5568; --text:#e2e8f0; }
html, body, [data-testid="stAppViewContainer"] { background-color:var(--bg) !important;
    font-family:'IBM Plex Sans',sans-serif; color:var(--text); }
[data-testid="stSidebar"] { background-color:var(--surface) !important; border-right:1px solid var(--border); }
h1,h2,h3 { font-family:'IBM Plex Mono',monospace; }
.metric-box { background:var(--surface); border:1px solid var(--border); border-top:2px solid var(--accent);
              padding:1.25rem; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

render_navigation(current_page="overview", show_quick_links=True)

st.markdown("""
<h2 style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; margin-bottom:0.25rem;">
◈ OVERVIEW
</h2>
<div style="color:#4a5568; font-size:0.8rem; letter-spacing:0.15em; margin-bottom:2rem;">
FRAUD STATISTICS & CLAIM DISTRIBUTIONS
</div>
""", unsafe_allow_html=True)

df = load_decisions()

if df.empty:
    st.warning("No processed data found. Run the pipeline first (Layers 1–3).")
    st.stop()

decision_col = "final_decision" if "final_decision" in df.columns else ("agent_action" if "agent_action" in df.columns else None)
state_col = "markov_state" if "markov_state" in df.columns else ("hmm_state" if "hmm_state" in df.columns else None)

stats = get_summary_stats(df)

# KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("TOTAL CLAIMS", f"{stats.get('total_claims', 0):,}", "#00e5ff"),
    ("HUMAN REVIEW", f"{stats.get('human_review', 0):,}", "#ff6b35"),
    ("HIGH RISK", f"{stats.get('high_risk', 0):,}", "#ff2d55"),
    ("AVG FRAUD PROB", f"{stats.get('avg_fraud_prob', 0):.1%}", "#ff9500"),
    ("REVIEW RATE", f"{stats.get('human_review',0)/max(stats.get('total_claims',1),1):.1%}", "#4a90d9"),
]
for col, (label, val, color) in zip([c1,c2,c3,c4,c5], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-box" style="border-top-color:{color};">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        color:#4a5568; letter-spacing:0.15em; margin-bottom:0.5rem;">{label}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
                        color:{color}; font-weight:600;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# Charts Row 1
col_left, col_right = st.columns([1, 1])

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0c0f", plot_bgcolor="#0a0c0f",
    font=dict(family="IBM Plex Mono", color="#8892a0", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)

with col_left:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>FRAUD PROBABILITY DISTRIBUTION</div>", unsafe_allow_html=True)
    if "fraud_probability" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["fraud_probability"],
            nbinsx=40,
            marker_color="#00e5ff",
            marker_line_color="#0a0c0f",
            marker_line_width=1,
            opacity=0.85,
            name="Claims"
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#ff2d55", line_width=1,
                      annotation_text="0.5 threshold", annotation_font_color="#ff2d55",
                      annotation_font_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title=None,
                          xaxis=dict(gridcolor="#1e2530", title="Fraud Probability"),
                          yaxis=dict(gridcolor="#1e2530", title="Claims"))
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>DECISION BREAKDOWN</div>", unsafe_allow_html=True)
    if decision_col:
        dec_counts = df[decision_col].value_counts()
        colors_map = {
            "approve":"#00ff88","fast_track":"#00e5ff","standard":"#ff6b35",
            "deep":"#ff9500","deny":"#ff2d55","fraud_detected":"#cc0033"
        }
        bar_colors = [colors_map.get(d, "#4a5568") for d in dec_counts.index]
        fig = go.Figure(go.Bar(
            x=dec_counts.index.str.upper().str.replace("_", " "),
            y=dec_counts.values,
            marker_color=bar_colors,
            marker_line_color="#0a0c0f",
            marker_line_width=1,
        ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          xaxis=dict(gridcolor="#1e2530"),
                          yaxis=dict(gridcolor="#1e2530", title="Count"))
        st.plotly_chart(fig, use_container_width=True)

# Charts Row 2
col_left2, col_right2 = st.columns([1, 1])

with col_left2:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>MARKOV STATE DISTRIBUTION</div>", unsafe_allow_html=True)
    if state_col:
        markov_counts = df[state_col].value_counts()
        fig = go.Figure(go.Pie(
            labels=markov_counts.index,
            values=markov_counts.values,
            hole=0.6,
            marker=dict(colors=["#00e5ff","#00ff88","#ff6b35","#ff2d55","#ff9500","#4a90d9"],
                        line=dict(color="#0a0c0f", width=2)),
            textfont=dict(family="IBM Plex Mono", size=10, color="#e2e8f0"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          legend=dict(font=dict(family="IBM Plex Mono", size=10)))
        st.plotly_chart(fig, use_container_width=True)

with col_right2:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>RISK TIER BREAKDOWN</div>", unsafe_allow_html=True)
    if "fraud_probability" in df.columns:
        bins = [0, 0.5, 1.001]
        labels = ["Low (<50%)", "High (\u226550%)"]
        _tier_series = pd.cut(
            df["fraud_probability"], bins=bins, labels=labels, include_lowest=True
        )
        tier_counts = _tier_series.value_counts().reindex(labels, fill_value=0)
        tier_colors = ["#00ff88", "#ff2d55"]
        fig = go.Figure(go.Bar(
            y=tier_counts.index,
            x=tier_counts.values,
            orientation="h",
            marker_color=tier_colors,
            marker_line_color="#0a0c0f",
            marker_line_width=1,
        ))
        fig.update_layout(**PLOTLY_LAYOUT,
                          xaxis=dict(gridcolor="#1e2530", title="Claims"),
                          yaxis=dict(gridcolor="#1e2530"))
        st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1.5rem 0 0.75rem 0;'>RECENT CLAIMS SAMPLE</div>", unsafe_allow_html=True)
display_cols = [c for c in ["claim_id", "fraud_probability"] if c in df.columns]
if decision_col and decision_col not in display_cols:
    display_cols.append(decision_col)
if state_col and state_col not in display_cols:
    display_cols.append(state_col)
if "requires_human_review" in df.columns:
    display_cols.append("requires_human_review")
st.dataframe(df[display_cols].head(20), use_container_width=True,
             hide_index=True,
             column_config={
                 "fraud_probability": st.column_config.ProgressColumn("Fraud Prob", min_value=0, max_value=1, format="%.2f"),
             })