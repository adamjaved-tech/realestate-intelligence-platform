"""Inference core shared by the REST API, the streaming scorer and batch scoring."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from propintel import config
from propintel.model.features import add_calendar, to_matrix

ONLINE_COLS = ["area_ppsf_lag1", "area_ppsf_roll3", "area_txn_lag1", "area_mom_growth", "city_ppsf_lag1"]
REQUEST_COLS = ["area", "property_type", "bedrooms", "bathrooms", "size_sqft", "floor",
                "age_years", "view", "furnished", "parking_spots"]


class Predictor:
    def __init__(self, bundle_path: Path | None = None, online_db: Path | None = None):
        self.bundle = joblib.load(bundle_path or (config.MODELS_DIR / "model_bundle.joblib"))
        self.online_db = online_db or config.ONLINE_DB
        self.online = self._load_online()

    @property
    def version(self) -> str:
        return self.bundle["version"]

    def _load_online(self) -> pd.DataFrame:
        with sqlite3.connect(self.online_db) as cx:
            return pd.read_sql("SELECT * FROM online_area_features", cx).set_index("area")

    def refresh_online(self) -> None:
        self.online = self._load_online()

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join request rows with the latest online market features."""
        unknown = sorted(set(df["area"]) - set(self.online.index))
        if unknown:
            raise ValueError(f"unknown area: {unknown}")
        out = df.merge(self.online[ONLINE_COLS + ["as_of_month"]], left_on="area", right_index=True, how="left")
        out["month"] = pd.to_datetime(out["as_of_month"])
        return add_calendar(out)

    def predict_frame(self, requests: pd.DataFrame) -> pd.DataFrame:
        df = self.enrich(requests[REQUEST_COLS].reset_index(drop=True))
        x = to_matrix(df)
        mid = np.exp(self.bundle["median"].predict(x))
        lo = np.minimum(np.exp(self.bundle["lower"].predict(x)), mid)
        hi = np.maximum(np.exp(self.bundle["upper"].predict(x)), mid)
        df["predicted_price_aed"] = mid.round(-2)
        df["lower_aed"] = lo.round(-2)
        df["upper_aed"] = hi.round(-2)
        df["predicted_ppsf"] = (mid / df["size_sqft"]).round(1)
        df["pred_to_baseline_ratio"] = mid / (df["size_sqft"] * df["area_ppsf_lag1"])
        return df

    def predict(self, records: list[dict]) -> list[dict]:
        df = self.predict_frame(pd.DataFrame(records))
        cols = ["predicted_price_aed", "lower_aed", "upper_aed", "predicted_ppsf"]
        res = df[cols].to_dict("records")
        for r in res:
            r["model_version"] = self.version
        return res
