"""
Layer 5 Core - Dashboard Data
Loads and caches processed data from Layers 1-3
"""

import pandas as pd
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
API_KEY = "dev-key"
HEADERS = {"X-API-Key": API_KEY}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@st.cache_data(ttl=60)
def load_decisions() -> pd.DataFrame:
    """Load Layer 3 agent decisions"""
    path = DATA_DIR / "agent_decisions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_features() -> pd.DataFrame:
    """Load Layer 1 processed features"""
    path = DATA_DIR / "processed_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", headers=HEADERS, timeout=2)
        return r.status_code == 200
    except:
        return False


def api_predict(claim: Dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}/predict/", json=claim, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_batch(claims: list) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}/batch/", json={"claims": claims}, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_explain(claim_id: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}/explain/{claim_id}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def get_summary_stats(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    stats = {
        "total_claims": len(df),
        "human_review": int(df.get("requires_human_review", pd.Series([False]*len(df))).sum()),
        "avg_fraud_prob": float(df["fraud_probability"].mean()) if "fraud_probability" in df.columns else 0,
        "high_risk": int((df["fraud_probability"] > 0.7).sum()) if "fraud_probability" in df.columns else 0,
    }
    if "final_decision" in df.columns:
        stats["decision_dist"] = df["final_decision"].value_counts().to_dict()
    if "markov_state" in df.columns:
        stats["markov_dist"] = df["markov_state"].value_counts().to_dict()
    return stats