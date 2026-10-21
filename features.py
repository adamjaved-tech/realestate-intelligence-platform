"""Single source of truth for turning rows into the model matrix (train == serve)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from propintel import config
from propintel.domain import AREAS, PROPERTY_TYPES, VIEWS

CATEGORIES = {
    "area": sorted(AREAS),
    "property_type": sorted(PROPERTY_TYPES),
    "view": sorted(VIEWS),
}


def add_calendar(df: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
    m = pd.to_datetime(df[month_col]).dt.month
    out = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    return out


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features, enforce dtypes and fixed categorical vocabularies."""
    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    x = df[config.MODEL_FEATURES].copy()
    for c in config.NUMERIC_FEATURES:
        x[c] = x[c].astype(float)
    for c, cats in CATEGORIES.items():
        bad = set(x[c].unique()) - set(cats)
        if bad:
            raise ValueError(f"unknown {c}: {sorted(bad)}")
        x[c] = pd.Categorical(x[c], categories=cats)
    return x


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / valid / test split (never random for time-dependent prices)."""
    months = sorted(pd.to_datetime(df["month"]).unique())
    test_start = months[-config.TEST_MONTHS]
    valid_start = months[-(config.TEST_MONTHS + config.VALID_MONTHS)]
    m = pd.to_datetime(df["month"])
    return df[m < valid_start], df[(m >= valid_start) & (m < test_start)], df[m >= test_start]
