"""
Page 4: Agent Monitor
Human review queue + Markov state visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from layer5.core.dashboard_data import load_decisions
from layer5.streamlit.components.claim_card import claim_card, DECISION_COLORS

st.set_page_config(page_title="Agent Monitor — StochClaim", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root { --bg:#0a0c0f; --surface:#111418; --border:#1e2530; --accent:#00e5ff; --text:#e2e8f0; }
html, body, [data-testid="stAppViewContainer"] { background-color:var(--bg) !important;
    font-family:'IBM Plex Sans',sans-serif; color:var(--text); }
[data-testid="stSidebar"] { background-color:var(--surface) !important; border-right:1px solid var(--border); }
h1,h2,h3 { font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; margin-bottom:0.25rem;">
◆ AGENT MONITOR
</h2>
<div style="color:#4a5568; font-size:0.8rem; letter-spacing:0.15em; margin-bottom:2rem;">
HUMAN REVIEW QUEUE & MARKOV STATE VISUALIZATION
</div>
""", unsafe_allow_html=True)

df = load_decisions()

if df.empty:
    st.warning("No processed data found. Run the pipeline first.")
    st.stop()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0c0f", plot_bgcolor="#0a0c0f",
    font=dict(family="IBM Plex Mono", color="#8892a0", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)

tab1, tab2 = st.tabs(["⚠ REVIEW QUEUE", "◈ MARKOV STATES"])

with tab1:
    review_df = df[df.get("requires_human_review", pd.Series([False]*len(df))) == True] if "requires_human_review" in df.columns else df

    # Summary bar
    total = len(df)
    review_count = len(review_df)
    pct = review_count / max(total, 1) * 100

    st.markdown(f"""
    <div style="background:#111418; border:1px solid #1e2530; padding:1rem 1.5rem;
                border-radius:4px; margin-bottom:1.5rem; display:flex;
                justify-content:space-between; align-items:center;">
        <div>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem;
                         color:#ff6b35; font-weight:600;">{review_count:,}</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                         color:#4a5568; margin-left:0.5rem;">claims require human review</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#4a5568;">
            {pct:.1f}% of total {total:,} claims
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sort controls
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sort_by = st.selectbox("Sort by", ["Fraud Probability (High→Low)", "Fraud Probability (Low→High)", "Decision"])
    with sc2:
        max_show = st.slider("Show", 10, 200, 50)

    if "fraud_probability" in review_df.columns:
        if "High→Low" in sort_by:
            review_df = review_df.sort_values("fraud_probability", ascending=False)
        elif "Low→High" in sort_by:
            review_df = review_df.sort_values("fraud_probability", ascending=True)
        elif "Decision" in sort_by and "final_decision" in review_df.columns:
            review_df = review_df.sort_values("final_decision")

    st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;color:#4a5568;margin-bottom:0.75rem;'>Showing {min(max_show, len(review_df))} of {len(review_df):,} claims</div>", unsafe_allow_html=True)

    for _, row in review_df.head(max_show).iterrows():
        claim_card(row.to_dict())

    # Download review queue
    csv = review_df.to_csv(index=False)
    st.download_button("⬇ DOWNLOAD REVIEW QUEUE CSV", csv,
                       file_name="human_review_queue.csv", mime="text/csv")

with tab2:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:1rem;'>MARKOV STATE FLOW</div>", unsafe_allow_html=True)

    if "markov_state" in df.columns:
        markov_counts = df["markov_state"].value_counts()

        # Sankey: decision → markov state
        if "final_decision" in df.columns:
            flow = df.groupby(["final_decision", "markov_state"]).size().reset_index(name="count")
            decisions = flow["final_decision"].unique().tolist()
            states = flow["markov_state"].unique().tolist()
            all_nodes = decisions + states
            node_colors = []
            for n in all_nodes:
                c = DECISION_COLORS.get(n.lower(), "#4a5568")
                node_colors.append(c)

            source = [all_nodes.index(d) for d in flow["final_decision"]]
            target = [all_nodes.index(s) for s in flow["markov_state"]]
            value = flow["count"].tolist()

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    line=dict(color="#0a0c0f", width=0.5),
                    label=[n.upper().replace("_", " ") for n in all_nodes],
                    color=node_colors,
                ),
                link=dict(source=source, target=target, value=value,
                          color=["#00e5ff22"] * len(source))
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=500)
            st.plotly_chart(fig, use_container_width=True)

        # State distribution heatmap by fraud prob bucket
        st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1.5rem 0 0.75rem 0;'>STATE vs FRAUD PROBABILITY</div>", unsafe_allow_html=True)
        if "fraud_probability" in df.columns:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
            df["prob_bucket"] = pd.cut(df["fraud_probability"], bins=bins, labels=labels)
            heatmap_data = df.groupby(["markov_state", "prob_bucket"]).size().unstack(fill_value=0)
            fig2 = go.Figure(go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.astype(str),
                y=heatmap_data.index,
                colorscale=[[0, "#0a0c0f"], [0.5, "#00e5ff44"], [1, "#00e5ff"]],
                showscale=True,
                text=heatmap_data.values,
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=10, color="#e2e8f0"),
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=350,
                               xaxis=dict(title="Fraud Probability Bucket"),
                               yaxis=dict(title=""))
            st.plotly_chart(fig2, use_container_width=True)

        # State table
        st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1rem 0 0.5rem 0;'>STATE STATISTICS</div>", unsafe_allow_html=True)
        state_stats = df.groupby("markov_state").agg(
            count=("markov_state", "count"),
            avg_fraud_prob=("fraud_probability", "mean") if "fraud_probability" in df.columns else ("markov_state", "count"),
        ).reset_index()
        state_stats.columns = ["Markov State", "Count", "Avg Fraud Probability"]
        state_stats = state_stats.sort_values("Count", ascending=False)
        st.dataframe(state_stats, use_container_width=True, hide_index=True,
                     column_config={
                         "Avg Fraud Probability": st.column_config.ProgressColumn(
                             "Avg Fraud Prob", min_value=0, max_value=1, format="%.2f")
                     })
    else:
        st.info("No Markov state data found in processed results.")