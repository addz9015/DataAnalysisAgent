"""
StochClaim Layer 5 - Streamlit Dashboard
Main entry point
"""

import streamlit as st
from layer5.streamlit.components.navigation import render_navigation

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

render_navigation(current_page="home", show_quick_links=True)