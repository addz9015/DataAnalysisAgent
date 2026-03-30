"""
Page 2: Claim Explorer
Individual claim lookup by claim ID
"""

import streamlit as st
import html
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from layer5.core.dashboard_data import (
    load_decisions,
    get_human_review_record,
    record_human_review_state,
)
from layer5.streamlit.components.risk_gauge import risk_gauge
from layer5.streamlit.components.claim_card import claim_card, DECISION_COLORS
from layer5.streamlit.components.decision_timeline import decision_timeline
from layer5.streamlit.components.navigation import render_navigation


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True

    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "<na>"}


def _clean_text(value, default="") -> str:
    return default if _is_missing(value) else str(value).strip()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

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

render_navigation(current_page="explorer", show_quick_links=True)

st.markdown("""
<h2 style="font-family:'IBM Plex Mono',monospace; letter-spacing:0.1em; margin-bottom:0.25rem;">
◎ CLAIM EXPLORER
</h2>
<div style="color:#4a5568; font-size:0.8rem; letter-spacing:0.15em; margin-bottom:2rem;">
INDIVIDUAL CLAIM LOOKUP & INSPECTION
</div>
""", unsafe_allow_html=True)

feedback = st.session_state.pop("claim_explorer_review_feedback", "")
if feedback:
    st.success(feedback)

# Force refresh so manual review CSV updates are reflected immediately in search results.
if hasattr(load_decisions, "clear"):
    load_decisions.clear()
df = load_decisions()
decision_col = "final_decision" if "final_decision" in df.columns else ("agent_action" if "agent_action" in df.columns else None)

if df.empty:
    st.warning("No processed data found. Run the pipeline first.")
    st.stop()

# Search
col_search, col_btn = st.columns([3, 1])
with col_search:
    search_id = st.text_input("Search Claim ID", placeholder="Enter claim ID  (e.g. 791425)", label_visibility="collapsed")
with col_btn:
    search_btn = st.button("SEARCH →", use_container_width=True)

# Show browse list if no search
if not search_id:
    st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin:1.5rem 0 0.75rem 0;'>BROWSE ALL CLAIMS</div>", unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        if decision_col:
            cleaned_decisions = sorted(
                {str(v).strip() for v in df[decision_col].tolist() if not _is_missing(v)}
            )
            decision_options = ["All"] + cleaned_decisions
        else:
            decision_options = ["All"]
        decision_filter = st.selectbox("Decision", decision_options)
    with fc2:
        review_filter = st.selectbox("Review Status", ["All", "Needs Review", "In Progress", "Cleared"])
    with fc3:
        risk_filter = st.selectbox("Risk Level", ["All", "Low (<50%)", "High (\u226550%)"])

    filtered = df.copy()
    if decision_filter != "All" and decision_col:
        filtered = filtered[
            filtered[decision_col].astype(str).str.strip().str.lower() == decision_filter.lower()
        ]
    if review_filter == "Needs Review" and "requires_human_review" in df.columns:
        filtered = filtered[filtered["requires_human_review"] == True]
    elif review_filter == "In Progress" and "human_review_status" in df.columns:
        filtered = filtered[
            filtered["human_review_status"].astype(str).str.strip().str.lower() == "in_progress"
        ]
    elif review_filter == "Cleared" and "requires_human_review" in df.columns:
        if "human_review_status" in filtered.columns:
            filtered = filtered[
                filtered["human_review_status"].astype(str).str.strip().str.lower().isin({"approved", "denied", "cleared"})
            ]
        else:
            filtered = filtered[filtered["requires_human_review"] == False]
    if risk_filter == "Low (<50%)" and "fraud_probability" in df.columns:
        filtered = filtered[filtered["fraud_probability"] < 0.5]
    elif risk_filter == "High (\u226550%)" and "fraud_probability" in df.columns:
        filtered = filtered[filtered["fraud_probability"] >= 0.5]

    if len(filtered) > 0:
        show_max = min(200, len(filtered))
        show_min = 10 if show_max >= 10 else 1
        show_default = min(50, show_max)
        if show_default < show_min:
            show_default = show_min

        slider_key = "claim_explorer_show_count"
        if slider_key in st.session_state:
            st.session_state[slider_key] = max(
                show_min,
                min(int(st.session_state[slider_key]), show_max),
            )

        show_count = st.slider(
            "Show",
            min_value=show_min,
            max_value=show_max,
            value=show_default,
            key=slider_key,
        )
    else:
        show_count = 0

    st.markdown(
        f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;color:#4a5568;margin-bottom:0.75rem;'>"
        f"Showing {min(show_count, len(filtered))} of {len(filtered):,} claims"
        "</div>",
        unsafe_allow_html=True,
    )
    for _, row in filtered.head(show_count).iterrows():
        claim_card(row.to_dict())

else:
    # Find claim
    id_col = "claim_id" if "claim_id" in df.columns else df.columns[0]
    search_key = str(search_id).strip()
    normalized_ids = (
        df[id_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    matches = df[normalized_ids == search_key]

    if matches.empty:
        safe_search_id = html.escape(str(search_id))
        st.markdown(f"""
        <div style="background:#111418; border:1px solid #ff2d5544; border-left:3px solid #ff2d55;
                    padding:1.25rem; border-radius:4px;">
            <span style="font-family:'IBM Plex Mono',monospace; color:#ff2d55;">
                ✕ Claim #{safe_search_id} not found
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        claim = matches.iloc[0].to_dict()

        # Always source latest human-review state directly from processed/unprocessed claim files.
        review_record = get_human_review_record(search_key)
        if review_record:
            claim.update(review_record)

        review_decision = _clean_text(claim.get("human_review_decision"), "").lower()
        review_bucket = _clean_text(claim.get("human_review_bucket"), "").lower()
        review_status = _clean_text(claim.get("human_review_status"), "").lower()
        reviewed_at = _clean_text(claim.get("reviewed_at_utc"), "")

        if review_status == "approved" or review_decision == "yes" or review_bucket == "processed":
            claim["requires_human_review"] = False
            claim["agent_action"] = "approve"
            claim["final_decision"] = "approve"
            claim["human_review_status"] = "CLEARED"
        elif review_status == "denied" or review_decision == "no" or review_bucket == "unprocessed":
            claim["requires_human_review"] = False
            claim["agent_action"] = "deny"
            claim["final_decision"] = "deny"
            claim["human_review_status"] = "REJECTED"
        elif review_status == "in_progress" or review_decision == "in_progress" or review_bucket == "in_progress":
            claim["requires_human_review"] = True
            claim["human_review_status"] = "IN_PROGRESS"

        fraud_prob = float(claim.get("fraud_probability", 0))
        decision = str(
            claim.get("final_decision")
            or claim.get("agent_action")
            or claim.get("agent_decision")
            or "unknown"
        ).lower()
        color = DECISION_COLORS.get(decision, "#4a5568")
        safe_search_id = html.escape(str(search_id))

        st.markdown(f"""
        <div style="background:#111418; border:1px solid {color}44; border-left:3px solid {color};
                    padding:1rem 1.5rem; border-radius:4px; margin-bottom:1.5rem;">
            <span style="font-family:'IBM Plex Mono',monospace; color:{color}; font-size:1.1rem;">
            Claim #{safe_search_id}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if review_decision == "yes" or review_bucket == "processed":
            reviewed_suffix = ""
            if reviewed_at:
                reviewed_suffix = f" • Reviewed at {html.escape(reviewed_at)}"
            st.markdown(
                f"""
                <div style="background:#0f2419;border:1px solid #00ff8844;border-left:3px solid #00ff88;
                            padding:0.8rem 1rem;border-radius:4px;margin:-0.5rem 0 1rem 0;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#00ff88;letter-spacing:0.1em;">
                        HUMAN REVIEW CLEARANCE
                    </div>
                    <div style="font-size:0.84rem;color:#9be8c0;margin-top:0.25rem;">
                        This claim was cleared by a human reviewer and moved to processed claims{reviewed_suffix}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif review_decision == "no" or review_bucket == "unprocessed":
            reviewed_suffix = ""
            if reviewed_at:
                reviewed_suffix = f" • Reviewed at {html.escape(reviewed_at)}"
            st.markdown(
                f"""
                <div style="background:#2a1418;border:1px solid #ff6b3544;border-left:3px solid #ff6b35;
                            padding:0.8rem 1rem;border-radius:4px;margin:-0.5rem 0 1rem 0;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#ffb084;letter-spacing:0.1em;">
                        HUMAN REVIEW REJECTED
                    </div>
                    <div style="font-size:0.84rem;color:#ffd7c2;margin-top:0.25rem;">
                        This claim was marked unprocessed by a human reviewer{reviewed_suffix}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif review_status == "in_progress" or review_decision == "in_progress" or review_bucket == "in_progress":
            reviewed_suffix = ""
            if reviewed_at:
                reviewed_suffix = f" • Updated at {html.escape(reviewed_at)}"
            st.markdown(
                f"""
                <div style="background:#2a240f;border:1px solid #ffd16644;border-left:3px solid #ffd166;
                            padding:0.8rem 1rem;border-radius:4px;margin:-0.5rem 0 1rem 0;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#ffd166;letter-spacing:0.1em;">
                        HUMAN REVIEW IN-PROGRESS
                    </div>
                    <div style="font-size:0.84rem;color:#ffe4a3;margin-top:0.25rem;">
                        This claim is currently under active human review and is in a temporary in-progress state{reviewed_suffix}.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if review_decision == "yes" or review_bucket == "processed":
            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-family:IBM Plex Mono;font-size:0.72rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.45rem;'>REVIEW OVERRIDE ACTIONS</div>",
                unsafe_allow_html=True,
            )
            c_reopen, c_reject, _ = st.columns([1, 1, 4])
            with c_reopen:
                reopen_clicked = st.button(
                    "↺ RETURN TO REVIEW",
                    key=f"claim_explorer_reopen_{search_key}",
                    use_container_width=True,
                    help="Undo clearance and move this claim back into human review",
                )
            with c_reject:
                reject_clicked = st.button(
                    "✕ REJECT CLAIM",
                    key=f"claim_explorer_reject_{search_key}",
                    use_container_width=True,
                    help="Mark this claim as rejected and move it to unprocessed claims",
                )

            if reopen_clicked:
                if record_human_review_state(claim, status="in_progress"):
                    st.session_state["claim_explorer_review_feedback"] = (
                        f"✓ Claim #{search_key} moved back to human review (in-progress)."
                    )
                    st.rerun()
                st.error("Could not move claim back to review because claim_id is missing.")

            if reject_clicked:
                if record_human_review_state(claim, status="denied"):
                    st.session_state["claim_explorer_review_feedback"] = (
                        f"✓ Claim #{search_key} marked as rejected (unprocessed)."
                    )
                    st.rerun()
                st.error("Could not reject claim because claim_id is missing.")

        r1, r2 = st.columns([1, 2])
        with r1:
            risk_gauge(fraud_prob)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            decision_timeline(claim)

        with r2:
            st.markdown("<div style='font-family:IBM Plex Mono;font-size:0.75rem;color:#4a5568;letter-spacing:0.15em;margin-bottom:0.75rem;'>CLAIM DETAILS</div>", unsafe_allow_html=True)
            excluded = {
                "_ingestion_timestamp",
                "_source_type",
                "_batch_id",
                "human_review_note",
                "explanation",
            }
            display_fields = {
                k: v
                for k, v in claim.items()
                if k not in excluded and not str(k).startswith("_") and not _is_missing(v)
            }
            cols = st.columns(2)
            items = list(display_fields.items())
            for i, (k, v) in enumerate(items):
                safe_key = html.escape(str(k).upper().replace("_", " "))
                safe_val = html.escape(_clean_text(v, "-")).replace("\n", "<br>")
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="padding:0.5rem 0; border-bottom:1px solid #1e2530;">
                        <div style="font-size:0.65rem; color:#4a5568; letter-spacing:0.1em;">{safe_key}</div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#e2e8f0; margin-top:0.15rem;">{safe_val}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Explanation
        explanation_text = _clean_text(claim.get("explanation"), "") or _clean_text(
            claim.get("explanation_summary"), ""
        )
        if explanation_text:
            safe_explanation = html.escape(explanation_text).replace("\n", "<br>")
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#111418; border:1px solid #1e2530; padding:1.25rem; border-radius:4px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                            color:#4a5568; letter-spacing:0.15em; margin-bottom:0.5rem;">AI EXPLANATION</div>
                <div style="font-size:0.9rem; color:#e2e8f0; line-height:1.6;">{safe_explanation}</div>
            </div>
            """, unsafe_allow_html=True)

        review_note = _clean_text(claim.get("human_review_note"), "")
        if not review_note and _as_bool(claim.get("requires_human_review")):
            review_note = _clean_text(claim.get("anomalies"), "") or _clean_text(
                claim.get("agent_reasoning"), ""
            )
        if review_note and _as_bool(claim.get("requires_human_review")):
            safe_review_note = html.escape(review_note.replace("|", "; ")).replace(
                "\n", "<br>"
            )
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div style="background:#2a1418;border:1px solid #ff2d5544;border-left:3px solid #ff6b35;'
                'padding:1.0rem 1.25rem;border-radius:4px;">'
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
                'color:#ffb3b3;letter-spacing:0.12em;margin-bottom:0.45rem;">HUMAN REVIEW NOTE</div>'
                f'<div style="font-size:0.88rem;color:#ffd7d7;line-height:1.55;">{safe_review_note}</div>'
                "</div>",
                unsafe_allow_html=True,
            )