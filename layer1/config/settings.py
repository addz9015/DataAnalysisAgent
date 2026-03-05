# layer1/config/settings.py
"""
Global settings and constants for Layer 1
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATION_LOGS_DIR = DATA_DIR / "validation_logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VALIDATION_LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data validation settings
VALIDATION_SETTINGS = {
    "strict_mode": False,           # If True, reject all invalid records
    "max_missing_ratio": 0.3,       # Reject if >30% fields missing
    "allowed_sources": ["csv", "json", "api", "dataframe"],
    "batch_size": 10000             # Process in chunks for large files
}

# Schema definitions (also in schema.yaml)
REQUIRED_COLUMNS = [
    "months_as_customer",
    "age", 
    "policy_annual_premium",
    "incident_severity",
    "total_claim_amount",
    "injury_claim",
    "property_claim", 
    "vehicle_claim",
    "incident_type",
    "collision_type",
    "authorities_contacted",
    "witness_present",
    "police_report_available"
]

OPTIONAL_COLUMNS = [
    "claim_id",
    "incident_date",
    "fraud_reported",
    "claim_status",
    "settlement_amount"
]

# Feature engineering settings
FEATURE_SETTINGS = {
    "claim_premium_threshold": 10.0,    # Flag if claim > 10x premium
    "high_risk_red_flags": 3,           # Threshold for high risk
    "severity_mapping": {
        "Trivial Damage": 1,
        "Minor Damage": 2,
        "Major Damage": 3,
        "Total Loss": 4
    },
    "tenure_bins": [0, 12, 36, 60, 600],
    "tenure_labels": ["New", "Short", "Medium", "Long"]
}

# Markov state definitions
MARKOV_STATES = {
    "transient": [
        "Fast_Track",
        "Complex_Review", 
        "Standard_Investigation",
        "High_Risk_Investigation"
    ],
    "absorbing": [
        "Approved",
        "Denied", 
        "Fraud_Detected"
    ]
}

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": str(DATA_DIR / "layer1.log"),
            "formatter": "detailed"
        }
    },
    "loggers": {
        "layer1": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}