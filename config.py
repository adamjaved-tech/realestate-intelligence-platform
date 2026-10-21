"""Central configuration: paths, feature contract, training splits."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PROPINTEL_HOME", Path(__file__).resolve().parents[2])).resolve()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
ONLINE_DIR = DATA_DIR / "online"
ONLINE_DB = ONLINE_DIR / "features.db"
MONITOR_DIR = DATA_DIR / "monitoring"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
KNOWLEDGE_DIR = ROOT / "docs" / "knowledge"

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "property-valuation"
MODEL_NAME = "property-valuation"

# ---- Feature contract (shared by training, batch scoring, API and streaming) ----
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "size_sqft", "floor", "age_years", "parking_spots", "furnished",
    "area_ppsf_lag1", "area_ppsf_roll3", "area_txn_lag1", "area_mom_growth", "city_ppsf_lag1",
    "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = ["area", "property_type", "view"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Property-level (request) features used for drift monitoring. Market-level features are
# excluded because they legitimately move every month.
DRIFT_NUMERIC = ["bedrooms", "bathrooms", "size_sqft", "floor", "age_years", "parking_spots",
                 "pred_to_baseline_ratio"]
DRIFT_CATEGORICAL = ["area", "property_type", "view"]
TARGET = "price_aed"

# ---- Training / evaluation ----
TEST_MONTHS = 6
VALID_MONTHS = 3
QUANTILES = (0.1, 0.9)
MAX_REJECT_RATE = 0.10          # data-quality gate for the silver layer
PSI_WARN, PSI_ALERT = 0.10, 0.25
