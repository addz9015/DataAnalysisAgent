"""Shared dashboard navigation shell (sidebar + quick links)."""

import streamlit as st

PAGES = [
    ("home", "⬡", "Home", "Main dashboard", "app.py"),
    ("overview", "◈", "Overview", "Fraud stats & distributions", "pages/01_overview.py"),
    ("explorer", "◎", "Claim Explorer", "Lookup & inspect claims", "pages/02_claim_explorer.py"),
    ("analyzer", "◉", "Fraud Analyzer", "Batch upload & analysis", "pages/03_fraud_analyzer.py"),
    ("monitor", "◆", "Agent Monitor", "Review queue & Markov states", "pages/04_agent_monitor.py"),
]


def render_navigation(current_page: str, show_quick_links: bool = True) -> None:
    """Render a consistent sidebar and optional clickable nav cards."""
    st.markdown(
        """
<style>
/* Keep only custom sidebar links and hide Streamlit generated page tree */
[data-testid="stSidebarNav"] { display: none; }

/* Shared clickable cards in main content */
section.main div[data-testid="stPageLink"] {
    margin-bottom: 0.5rem;
}

section.main div[data-testid="stPageLink"] a,
section.main div[data-testid="stPageLink"] > div > a {
    background: #111418;
    border: 1px solid #1e2530;
    border-top: 2px solid #00e5ff;
    border-radius: 4px;
    min-height: 220px;
    padding: 1.25rem;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    width: 100%;
    text-decoration: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

section.main div[data-testid="stPageLink"] a:hover,
section.main div[data-testid="stPageLink"] > div > a:hover {
    border-color: #00e5ff;
    box-shadow: 0 0 0 1px #00e5ff inset;
}

section.main div[data-testid="stPageLink"] a p,
section.main div[data-testid="stPageLink"] > div > a p {
    white-space: pre-line;
    margin: 0;
    color: #00e5ff !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.55;
    letter-spacing: 0.05em;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
<div style="padding: 1rem 0 2rem 0; border-bottom: 1px solid #1e2530; margin-bottom: 1rem;">
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: #00e5ff; letter-spacing: 0.15em;">
        ⬡ STOCHCLAIM
    </div>
    <div style="font-size: 0.7rem; color: #4a5568; letter-spacing: 0.2em; margin-top: 0.25rem;">
        FRAUD INTELLIGENCE v1.0
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    for page_key, icon, label, _, target in PAGES:
        text = f"{icon}  {label}"
        if page_key == current_page:
            text = f"{icon}  {label}"
        st.sidebar.page_link(target, label=text, icon=None)

    if not show_quick_links:
        return

    cols = st.columns(4)
    quick_pages = PAGES[1:]
    for col, (page_key, icon, label, desc, target) in zip(cols, quick_pages):
        with col:
            display_label = f"{icon}  {label}\n{desc}"
            st.page_link(target, label=display_label, use_container_width=True)
