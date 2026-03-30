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
from layer5.core.dashboard_data import (
    load_decisions,
    get_human_reviewed_ids,
    record_human_review_decision,
    record_human_review_state,
)
from layer5.streamlit.components.claim_card import claim_card, DECISION_COLORS
from layer5.streamlit.components.navigation import render_navigation


def _normalize_claim_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _ensure_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Markov/HMM columns from Layer 2 output when missing in decisions."""
    if df.empty:
        return df

    if "markov_state" in df.columns or "hmm_state" in df.columns:
        return df

    if "claim_id" not in df.columns:
        return df

    layer2_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "layer2_results.csv"
    if not layer2_path.exists():
        return df

    state_cols = {"claim_id", "markov_state", "hmm_state"}
    layer2_df = pd.read_csv(layer2_path, usecols=lambda c: c in state_cols)
    if layer2_df.empty or "claim_id" not in layer2_df.columns:
        return df

    layer2_df = layer2_df.drop_duplicates(subset=["claim_id"], keep="last").copy()
    merged = df.copy()
    merged["claim_id"] = _normalize_claim_id(merged["claim_id"])
    layer2_df["claim_id"] = _normalize_claim_id(layer2_df["claim_id"])

    keep_cols = [c for c in ["claim_id", "markov_state", "hmm_state"] if c in layer2_df.columns]
    return merged.merge(layer2_df[keep_cols], on="claim_id", how="left")

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

render_navigation(current_page="monitor", show_quick_links=True)

st.markdown("""
<h2 style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; margin-bottom:0.25rem;">
◆ AGENT MONITOR
</h2>
<div style="color:#4a5568; font-size:0.8rem; letter-spacing:0.15em; margin-bottom:2rem;">
HUMAN REVIEW QUEUE & MARKOV STATE VISUALIZATION
</div>
""", unsafe_allow_html=True)

df = _ensure_state_columns(load_decisions())

if df.empty:
    st.warning("No processed data found. Run the pipeline first.")
    st.stop()

decision_col = "final_decision" if "final_decision" in df.columns else ("agent_action" if "agent_action" in df.columns else None)
state_col = "markov_state" if "markov_state" in df.columns else ("hmm_state" if "hmm_state" in df.columns else None)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0c0f", plot_bgcolor="#0a0c0f",
    font=dict(family="IBM Plex Mono", color="#8892a0", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
)

tab1, tab2 = st.tabs(["⚠ REVIEW QUEUE", "◈ MARKOV STATES"])

with tab1:
    review_df = df[df.get("requires_human_review", pd.Series([False]*len(df))) == True] if "requires_human_review" in df.columns else df

    if "human_review_status" in review_df.columns:
        review_df = review_df.copy()
        review_df["human_review_status"] = review_df["human_review_status"].fillna("pending").astype(str).str.lower()
    else:
        review_df = review_df.copy()
        review_df["human_review_status"] = "pending"

    if "claim_id" in review_df.columns:
        reviewed_ids = get_human_reviewed_ids()
        if reviewed_ids:
            current_ids = _normalize_claim_id(review_df["claim_id"])
            review_df = review_df[~current_ids.isin(reviewed_ids)].copy()

    in_progress_df = review_df[review_df["human_review_status"] == "in_progress"].copy()
    pending_df = review_df[review_df["human_review_status"] != "in_progress"].copy()

    feedback = st.session_state.pop("human_review_feedback", "")
    if feedback:
        st.success(feedback)

    # Summary bar
    total = len(df)
    review_count = len(pending_df)
    in_progress_count = len(in_progress_df)
    total_open = review_count + in_progress_count
    pct = review_count / max(total, 1) * 100

    st.markdown(f"""
    <div style="background:#111418; border:1px solid #1e2530; padding:1rem 1.5rem;
                border-radius:4px; margin-bottom:1.5rem; display:flex;
                justify-content:space-between; align-items:center;">
        <div>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem;
                         color:#ff6b35; font-weight:600;">{review_count:,}</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                         color:#4a5568; margin-left:0.5rem;">pending review</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                         color:#00e5ff; margin-left:1rem;">{in_progress_count:,} in progress</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#4a5568;">
            {total_open:,} open ({pct:.1f}% pending) of total {total:,} claims
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
            in_progress_df = in_progress_df.sort_values("fraud_probability", ascending=False)
            pending_df = pending_df.sort_values("fraud_probability", ascending=False)
        elif "Low→High" in sort_by:
            in_progress_df = in_progress_df.sort_values("fraud_probability", ascending=True)
            pending_df = pending_df.sort_values("fraud_probability", ascending=True)
        elif "Decision" in sort_by and decision_col and decision_col in review_df.columns:
            in_progress_df = in_progress_df.sort_values(decision_col)
            pending_df = pending_df.sort_values(decision_col)

    review_df = pd.concat([in_progress_df, pending_df], ignore_index=False)

    st.markdown(f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;color:#4a5568;margin-bottom:0.75rem;'>Showing {min(max_show, len(review_df))} of {len(review_df):,} open review claims</div>", unsafe_allow_html=True)

    for idx, row in review_df.head(max_show).iterrows():
        claim = row.to_dict()
        claim_card(claim)

        claim_id = str(claim.get("claim_id", idx)).strip()
        if claim_id.endswith(".0"):
            claim_id = claim_id[:-2]
        c_yes, c_no, c_progress, c_help = st.columns([1, 1, 1, 5])
        with c_yes:
            yes_clicked = st.button(
                "✓ YES",
                key=f"review_yes_{claim_id}_{idx}",
                use_container_width=True,
                help="Add this claim to processed_claims.csv",
            )
        with c_no:
            no_clicked = st.button(
                "✕ NO",
                key=f"review_no_{claim_id}_{idx}",
                use_container_width=True,
                help="Add this claim to unprocessed_claims.csv",
            )
        with c_progress:
            progress_clicked = st.button(
                "⧗ IN-PROGRESS",
                key=f"review_progress_{claim_id}_{idx}",
                use_container_width=True,
                help="Mark this claim as temporarily in-progress during human review",
            )
        with c_help:
            st.markdown(
                "<div style='font-family:IBM Plex Mono;font-size:0.68rem;color:#4a5568;padding-top:0.35rem;'>"
                "Human decision: YES=processed, NO=unprocessed, IN-PROGRESS=temporary review state."
                "</div>",
                unsafe_allow_html=True,
            )

        if yes_clicked:
            if record_human_review_decision(claim, approved=True):
                st.session_state["human_review_feedback"] = f"✓ Claim #{claim_id} added to processed_claims.csv"
                st.rerun()
            st.error("Could not store decision because claim_id is missing.")

        if no_clicked:
            if record_human_review_decision(claim, approved=False):
                st.session_state["human_review_feedback"] = f"✓ Claim #{claim_id} added to unprocessed_claims.csv"
                st.rerun()
            st.error("Could not store decision because claim_id is missing.")

        if progress_clicked:
            if record_human_review_state(claim, status="in_progress"):
                st.session_state["human_review_feedback"] = f"✓ Claim #{claim_id} marked as in-progress"
                st.rerun()
            st.error("Could not set in-progress state because claim_id is missing.")

    # Download review queue
    csv = review_df.to_csv(index=False)
    st.download_button("⬇ DOWNLOAD REVIEW QUEUE CSV", csv,
                       file_name="human_review_queue.csv", mime="text/csv")

with tab2:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:1rem;'>MARKOV STATE FLOW</div>", unsafe_allow_html=True)

    if state_col:
        markov_counts = df[state_col].value_counts()

        # Sankey: decision → markov state
        if decision_col:
            flow = df.groupby([decision_col, state_col]).size().reset_index(name="count")
            decisions = flow[decision_col].unique().tolist()
            states = flow[state_col].unique().tolist()
            all_nodes = decisions + states
            node_colors = []
            for n in all_nodes:
                c = DECISION_COLORS.get(n.lower(), "#4a5568")
                node_colors.append(c)

            source = [all_nodes.index(d) for d in flow[decision_col]]
            target = [all_nodes.index(s) for s in flow[state_col]]
            value = flow["count"].tolist()

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    line=dict(color="#0a0c0f", width=0.5),
                    label=[n.upper().replace("_", " ") for n in all_nodes],
                    color=node_colors,
                ),
                link=dict(source=source, target=target, value=value,
                          color=["rgba(0,229,255,0.13)"] * len(source))
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=500)
            st.plotly_chart(fig, use_container_width=True)

        # State distribution heatmap by fraud prob bucket
        st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1.5rem 0 0.75rem 0;'>STATE vs FRAUD PROBABILITY</div>", unsafe_allow_html=True)
        if "fraud_probability" in df.columns:
            bins = [0, 0.5, 1.001]
            labels = ["Low (<50%)", "High (\u226550%)"]
            _bucket_series = pd.cut(
                df["fraud_probability"], bins=bins, labels=labels, include_lowest=True
            )
            _bucket_df = df[[state_col]].copy()
            _bucket_df["prob_bucket"] = _bucket_series
            heatmap_data = _bucket_df.groupby([state_col, "prob_bucket"]).size().unstack(fill_value=0)
            fig2 = go.Figure(go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.astype(str),
                y=heatmap_data.index,
                colorscale=[[0, "#0a0c0f"], [0.5, "rgba(0,229,255,0.27)"], [1, "#00e5ff"]],
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
        state_stats = df.groupby(state_col).agg(
            count=(state_col, "count"),
            avg_fraud_prob=("fraud_probability", "mean") if "fraud_probability" in df.columns else (state_col, "count"),
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