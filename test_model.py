import json

import pandas as pd
import pytest

from propintel import config
from propintel.model.features import time_split, to_matrix
from propintel.model.predictor import Predictor

PROP = {"area": "Dubai Marina", "property_type": "Apartment", "bedrooms": 2, "bathrooms": 3,
        "size_sqft": 1250.0, "floor": 22, "age_years": 4, "view": "Sea", "furnished": 0, "parking_spots": 1}


def test_model_beats_naive_baseline(built):
    t = built["train"]
    assert t["test_mape"] < t["baseline_mape"]
    assert t["test_r2"] > 0.9
    assert abs(t["interval_coverage"] - t["interval_nominal"]) < 0.15
    assert t["promoted"] in (True, False)
    assert json.loads((config.REPORTS_DIR / "metrics.json").read_text())["version"] == t["version"]


def test_time_split_is_chronological(built):
    df = pd.read_parquet(config.GOLD_DIR / "features.parquet")
    tr, va, te = time_split(df)
    assert tr["month"].max() < va["month"].min() <= va["month"].max() < te["month"].min()


def test_to_matrix_rejects_unknown_category(built):
    df = pd.read_parquet(config.GOLD_DIR / "features.parquet").head(3).copy()
    from propintel.model.features import add_calendar
    df = add_calendar(df)
    df.loc[df.index[0], "area"] = "Atlantis"
    with pytest.raises(ValueError):
        to_matrix(df)


def test_predictor_outputs_are_sane_and_ordered(built):
    p = Predictor()
    r = p.predict([PROP])[0]
    assert r["lower_aed"] <= r["predicted_price_aed"] <= r["upper_aed"]
    assert 1_000_000 < r["predicted_price_aed"] < 20_000_000
    bigger = p.predict([{**PROP, "size_sqft": 2500.0}])[0]
    assert bigger["predicted_price_aed"] > r["predicted_price_aed"]
    with pytest.raises(ValueError):
        p.predict([{**PROP, "area": "Atlantis"}])


def test_mlflow_registered_model_with_alias(built):
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    c = MlflowClient()
    versions = c.search_model_versions(f"name='{config.MODEL_NAME}'")
    assert len(versions) >= 1
    if built["train"]["promoted"]:
        assert c.get_model_version_by_alias(config.MODEL_NAME, "champion")
