"""Training: LightGBM valuation model + quantile interval models, tracked in MLflow.

* target = log(price) (multiplicative price structure, scale-free errors)
* chronological split, early stopping on the validation window
* baseline = size x last-month area median price/sqft (what a naive analyst would do)
* promotion gate: model gets the `champion` alias only if it beats the baseline
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from propintel import config
from propintel.model.features import add_calendar, time_split, to_matrix

log = logging.getLogger("propintel.train")

PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
              n_jobs=-1, verbose=-1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ape = np.abs(y_pred - y_true) / y_true
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae_aed": float(np.mean(np.abs(y_pred - y_true))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "r2": float(1 - ss_res / ss_tot),
    }


def train(n_estimators: int = 2000, register: bool = True) -> dict:
    t0 = time.time()
    df = pd.read_parquet(config.GOLD_DIR / "features.parquet")
    df = add_calendar(df)
    tr, va, te = time_split(df)
    log.info("split sizes train=%d valid=%d test=%d", len(tr), len(va), len(te))
    xtr, xva, xte = to_matrix(tr), to_matrix(va), to_matrix(te)
    ytr, yva = np.log(tr[config.TARGET].values), np.log(va[config.TARGET].values)

    model = lgb.LGBMRegressor(n_estimators=n_estimators, objective="regression", **PARAMS)
    model.fit(xtr, ytr, eval_set=[(xva, yva)], eval_metric="l1",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = int(model.best_iteration_ or n_estimators)

    quant = {}
    for q in config.QUANTILES:
        m = lgb.LGBMRegressor(n_estimators=best_iter, objective="quantile", alpha=q, **PARAMS)
        m.fit(xtr, ytr)
        quant[q] = m

    # ---------------- evaluation on the untouched, most recent test window
    y_true = te[config.TARGET].values
    pred = np.exp(model.predict(xte))
    lo = np.exp(quant[config.QUANTILES[0]].predict(xte))
    hi = np.exp(quant[config.QUANTILES[1]].predict(xte))
    lo, hi = np.minimum(lo, pred), np.maximum(hi, pred)
    baseline = te["size_sqft"].values * te["area_ppsf_lag1"].values
    m_model, m_base = regression_metrics(y_true, pred), regression_metrics(y_true, baseline)
    coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
    nominal = config.QUANTILES[1] - config.QUANTILES[0]

    per_area = (te.assign(ape=np.abs(pred - y_true) / y_true)
                  .groupby("area")["ape"].agg(mape="mean", n="size").reset_index()
                  .sort_values("mape"))
    importance = (pd.DataFrame({"feature": config.MODEL_FEATURES,
                                "gain": model.booster_.feature_importance("gain")})
                    .sort_values("gain", ascending=False))

    # Reference sample for drift monitoring (property-level features + valuation ratio)
    ref = tr.sample(min(5000, len(tr)), random_state=0).copy()
    ref_pred = np.exp(model.predict(to_matrix(ref)))
    ref["pred_to_baseline_ratio"] = ref_pred / (ref["size_sqft"] * ref["area_ppsf_lag1"])

    metrics = {
        **{f"test_{k}": v for k, v in m_model.items()},
        **{f"baseline_{k}": v for k, v in m_base.items()},
        "interval_coverage": coverage, "interval_nominal": nominal, "best_iteration": best_iter,
        "train_rows": len(tr), "valid_rows": len(va), "test_rows": len(te),
    }
    gate = (m_model["mape"] < m_base["mape"]) and abs(coverage - nominal) < 0.08
    version = hashlib.sha1(f"{time.time()}-{metrics['test_mape']}".encode()).hexdigest()[:8]

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": version, "median": model, "lower": quant[config.QUANTILES[0]],
        "upper": quant[config.QUANTILES[1]], "features": config.MODEL_FEATURES,
        "metrics": metrics, "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "train_window": [str(tr["month"].min())[:10], str(tr["month"].max())[:10]],
        "test_window": [str(te["month"].min())[:10], str(te["month"].max())[:10]],
    }
    joblib.dump(bundle, config.MODELS_DIR / "model_bundle.joblib")
    ref.to_parquet(config.MODELS_DIR / "reference.parquet")
    per_area.to_csv(config.REPORTS_DIR / "per_area_metrics.csv", index=False)
    importance.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps({"version": version, **metrics, "promoted": gate}, indent=2))

    if register:
        _log_mlflow(model, metrics, best_iter, version, gate, per_area, importance, xte.head(5))

    log.info("trained %s in %.1fs | MAPE %.2f%% vs baseline %.2f%% | coverage %.1f%%",
             version, time.time() - t0, 100 * m_model["mape"], 100 * m_base["mape"], 100 * coverage)
    return {"version": version, "promoted": gate, **metrics}


def _log_mlflow(model, metrics, best_iter, version, gate, per_area, importance, sample) -> None:
    (config.ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"lgbm-{version}"):
        mlflow.log_params({**{k: v for k, v in PARAMS.items() if k != "verbose"},
                           "best_iteration": best_iter, "target": "log_price",
                           "n_features": len(config.MODEL_FEATURES), "bundle_version": version})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(config.REPORTS_DIR / "per_area_metrics.csv"))
        mlflow.log_artifact(str(config.REPORTS_DIR / "feature_importance.csv"))
        mlflow.log_artifact(str(config.REPORTS_DIR / "metrics.json"))
        info = mlflow.lightgbm.log_model(model.booster_, name="model",
                                        registered_model_name=config.MODEL_NAME)
        if gate:
            MlflowClient().set_registered_model_alias(
                config.MODEL_NAME, "champion", info.registered_model_version)
        mlflow.set_tag("promoted", str(gate))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(train(), indent=2))


if __name__ == "__main__":
    main()
