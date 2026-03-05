"""
StochClaim Layer 5 - Streamlit Dashboard
Main entry point
"""

import streamlit as st

st.set_page_config(
    page_title="StochClaim — Fraud Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0c0f;
    --surface:   #111418;
    --border:    #1e2530;
    --accent:    #00e5ff;
    --warn:      #ff6b35;
    --danger:    #ff2d55;
    --ok:        #00ff88;
    --muted:     #4a5568;
    --text:      #e2e8f0;
    --subtext:   #8892a0;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

.stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent);
    color: var(--bg);
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# Sidebar branding
st.sidebar.markdown("""
<div style="padding: 1rem 0 2rem 0; border-bottom: 1px solid #1e2530; margin-bottom: 1rem;">
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: #00e5ff; letter-spacing: 0.15em;">
        ⬡ STOCHCLAIM
    </div>
    <div style="font-size: 0.7rem; color: #4a5568; letter-spacing: 0.2em; margin-top: 0.25rem;">
        FRAUD INTELLIGENCE v1.0
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.page_link("app.py", label="⬡  Home", icon=None)
st.sidebar.page_link("pages/01_overview.py", label="◈  Overview", icon=None)
st.sidebar.page_link("pages/02_claim_explorer.py", label="◎  Claim Explorer", icon=None)
st.sidebar.page_link("pages/03_fraud_analyzer.py", label="◉  Fraud Analyzer", icon=None)
st.sidebar.page_link("pages/04_agent_monitor.py", label="◆  Agent Monitor", icon=None)

# Home page
st.markdown("""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:2rem; padding-top:1rem;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:2.5rem; color:#00e5ff;">⬡</div>
    <div>
        <h1 style="margin:0; font-size:1.8rem; letter-spacing:0.1em; color:#e2e8f0;">STOCHCLAIM</h1>
        <div style="color:#4a5568; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.2em;">
            INSURANCE FRAUD INTELLIGENCE PLATFORM
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
cards = [
    ("◈", "Overview", "Fraud stats & distributions", "pages/01_overview.py"),
    ("◎", "Claim Explorer", "Lookup & inspect claims", "pages/02_claim_explorer.py"),
    ("◉", "Fraud Analyzer", "Batch upload & analysis", "pages/03_fraud_analyzer.py"),
    ("◆", "Agent Monitor", "Review queue & Markov states", "pages/04_agent_monitor.py"),
]
for col, (icon, title, desc, page) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(f"""
        <div style="background:#111418; border:1px solid #1e2530; border-top:2px solid #00e5ff;
                    padding:1.5rem; border-radius:4px; min-height:140px;">
            <div style="font-size:1.5rem; margin-bottom:0.75rem;">{icon}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                        color:#00e5ff; margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.8rem; color:#4a5568;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page, label=f"Open {title} →")