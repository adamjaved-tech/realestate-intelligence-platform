"""Data / prediction drift detection (PSI + two-sample KS) and performance monitoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from propintel import config


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return _psi(r, c)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi(r, c)


def _psi(r: np.ndarray, c: np.ndarray, eps: float = 1e-4) -> float:
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(psi: float) -> str:
    return "ALERT" if psi >= config.PSI_ALERT else "WARN" if psi >= config.PSI_WARN else "OK"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, min_rows: int = 200) -> dict:
    if len(current) < min_rows:
        return {"status": "INSUFFICIENT_DATA", "rows": len(current), "min_rows": min_rows, "features": {}}
    feats = {}
    for c in config.DRIFT_NUMERIC:
        p = psi_numeric(reference[c].astype(float).values, current[c].astype(float).values)
        ks = stats.ks_2samp(reference[c].astype(float), current[c].astype(float))
        feats[c] = {"type": "numeric", "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                    "ks_pvalue": float(ks.pvalue), "status": status(p)}
    for c in config.DRIFT_CATEGORICAL:
        p = psi_categorical(reference[c], current[c])
        feats[c] = {"type": "categorical", "psi": round(p, 4), "status": status(p)}
    order = {"OK": 0, "WARN": 1, "ALERT": 2}
    overall = max((f["status"] for f in feats.values()), key=order.get)
    return {"status": overall, "rows": len(current),
            "drifted_features": [k for k, v in feats.items() if v["status"] != "OK"], "features": feats}


def performance_report(y_true: np.ndarray, y_pred: np.ndarray, baseline_mape: float, tolerance: float = 1.25) -> dict:
    """Once ground truth lands: alert if live MAPE degrades beyond `tolerance` x training-time MAPE."""
    mape = float(np.mean(np.abs(y_pred - y_true) / y_true))
    return {"live_mape": mape, "reference_mape": baseline_mape,
            "status": "ALERT" if mape > baseline_mape * tolerance else "OK"}


def write_report(report: dict, path: Path | None = None) -> Path:
    path = path or (config.REPORTS_DIR / "drift_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def load_logged_requests(path: Path | None = None) -> pd.DataFrame:
    path = path or (config.MONITOR_DIR / "predictions.jsonl")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)
