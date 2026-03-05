# StochClaim — Insurance Fraud Intelligence Platform

> A five-layer stochastic AI pipeline for automated insurance claim fraud detection, risk scoring, and intelligent routing.

---

## Overview

StochClaim is an end-to-end insurance fraud detection system that processes claims through five sequential layers — from raw data ingestion to an interactive dashboard. It combines probabilistic modelling, ensemble machine learning, Markov chain state transitions, and LLM-powered explanations to produce interpretable fraud decisions at scale.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STOCHCLAIM PIPELINE                      │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│ Layer 1  │ Layer 2  │ Layer 3  │ Layer 4  │ Layer 5            │
│ Data     │Stochastic│  Agent   │  FastAPI │ Streamlit          │
│ Intake   │ Ensemble │Orchestr. │   API    │ Dashboard          │
├──────────┼──────────┼──────────┼──────────┼────────────────────┤
│ Ingest   │ HMM +    │ Reasoning│ REST     │ Overview           │
│ Validate │ Gradient │ LLM      │ Endpoints│ Claim Explorer     │
│ Preproc. │ Boosting │ Decisions│ Auth     │ Fraud Analyzer     │
│ Features │ Markov   │ Routing  │ Rate     │ Agent Monitor      │
│          │ States   │          │ Limiting │                    │
└──────────┴──────────┴──────────┴──────────┴────────────────────┘
```

---

## Features

- **Multi-source data ingestion** — CSV, JSON, Excel, Parquet, and direct API input
- **Stochastic ensemble modelling** — Hidden Markov Models combined with gradient boosting
- **Markov chain state transitions** — Claims routed through transient and absorbing states
- **LLM-powered explanations** — Natural language decision explanations via Groq (llama-3.3-70b-versatile)
- **Intelligent claim routing** — Approve, Fast Track, Standard, Deep Investigation, or Deny
- **Human review queue** — Automatic flagging of high-uncertainty claims
- **REST API** — FastAPI backend with API key authentication and rate limiting
- **Interactive dashboard** — Streamlit frontend with real-time fraud analytics

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Python, Pandas, NumPy |
| ML Models | scikit-learn, Hidden Markov Models, Gradient Boosting |
| LLM Integration | Groq API (llama-3.3-70b-versatile) |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Data Storage | CSV, JSON, Parquet |

---

## Project Structure

```
stochclaim/
├── layer1/               # Data Intake & Preprocessing
│   ├── core/
│   │   ├── pipeline.py
│   │   ├── intake.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   └── feature_store.py
│   └── config/
│       └── settings.py
├── layer2/               # Stochastic Ensemble Models
│   └── core/
│       └── ensemble.py
├── layer3/               # Agent Orchestration & LLM
│   ├── core/
│   │   ├── agent_orchestrator.py
│   │   ├── reasoning_engine.py
│   │   └── action_selector.py
│   └── llm/
│       └── explainer_llm.py
├── layer4/               # FastAPI REST API
│   ├── routers/
│   ├── middleware/
│   └── config.py
├── layer5/               # Streamlit Dashboard
│   ├── streamlit/
│   │   ├── app.py
│   │   ├── pages/
│   │   │   ├── 01_overview.py
│   │   │   ├── 02_claim_explorer.py
│   │   │   ├── 03_fraud_analyzer.py
│   │   │   └── 04_agent_monitor.py
│   │   └── components/
│   └── core/
│       └── dashboard_data.py
└── data/
    ├── raw/
    └── processed/
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### Install Dependencies

```bash
# Core pipeline
pip install -r requirements.txt

# Dashboard
pip install -r layer5/requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
API_KEY=dev-key
```

---

## Running the Pipeline

### Step 1 — Process Claims Data (Layers 1–3)

```bash
python run_layer1.py
python run_layer2.py
python run_layer3.py
```

### Step 2 — Start the API (Layer 4)

```bash
# Terminal 1
python run_layer4.py
# API running at http://127.0.0.1:8000
```

### Step 3 — Launch the Dashboard (Layer 5)

```bash
# Terminal 2
python -m streamlit run layer5/streamlit/app.py
# Dashboard at http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| POST | `/predict/` | Single claim fraud prediction |
| POST | `/batch/` | Batch claim processing |
| POST | `/query/ask` | Natural language query |
| POST | `/query/quick-check` | Quick claim check |
| GET | `/explain/{id}` | Get claim explanation |
| POST | `/feedback/` | Submit correction feedback |

All endpoints require `X-API-Key` header.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | Fraud statistics, KPIs, claim distributions, Markov state breakdown |
| Claim Explorer | Search claims by ID, filter by decision/risk/review status |
| Fraud Analyzer | Upload CSV/JSON/Excel for batch analysis, manual single claim entry |
| Agent Monitor | Human review queue (660 flagged claims), Markov state Sankey diagram |

---

## Markov States

**Transient States** (claims in progress):
- `Fast_Track` — Low risk, quick approval path
- `Standard_Investigation` — Medium risk, standard review
- `Complex_Review` — High uncertainty, needs analysis
- `High_Risk_Investigation` — High fraud probability

**Absorbing States** (final outcomes):
- `Approved` — Claim approved for payment
- `Denied` — Claim denied
- `Fraud_Detected` — Confirmed fraud, escalated

---

## Results

On a dataset of 1,000 insurance claims:
- **660 claims** flagged for human review
- Processing time: ~4–6 seconds (without LLM), ~25 minutes (with LLM explanations)
- Supports CSV, JSON, Excel, and Parquet input formats

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
