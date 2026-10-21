import pandas as pd
import pytest

from propintel import config
from propintel.data.generate import generate_clean, inject_dirt
from propintel.pipeline import medallion


def test_generator_is_deterministic_and_realistic():
    a, b = generate_clean(500, seed=3), generate_clean(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a["price_aed"] > 0).all() and a["transaction_id"].is_unique
    assert 150 < (a["price_aed"] / a["size_sqft"]).median() < 10_000


def test_dirt_injection_adds_duplicates_and_nulls():
    d = inject_dirt(generate_clean(2000, seed=3))
    assert d["transaction_id"].duplicated().any() and d["size_sqft"].isna().any()


def test_silver_quarantines_bad_rows_and_dedupes(built):
    dq = built["pipeline"]["data_quality"]
    assert dq["passed"] and dq["duplicates_removed"] > 0 and dq["rows_quarantined"] > 0
    silver = pd.read_parquet(config.SILVER_DIR / "transactions.parquet")
    assert silver["transaction_id"].is_unique
    assert silver["area"].notna().all() and (silver["price_aed"] > 0).all()
    assert silver["ppsf"].between(200, 15000).all()
    q = pd.read_parquet(config.SILVER_DIR / "quarantine.parquet")
    assert q["reject_reason"].notna().all()
    assert dq["rows_valid"] + dq["rows_quarantined"] == dq["rows_in"] - dq["duplicates_removed"]


def test_quality_gate_fails_on_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_REJECT_RATE", 0.0001)
    with pytest.raises(medallion.DataQualityError):
        medallion.run_silver()


def test_gold_features_have_no_leakage(built):
    """Market features for month M must be computed only from months < M."""
    f = pd.read_parquet(config.GOLD_DIR / "features.parquet")
    silver = pd.read_parquet(config.SILVER_DIR / "transactions.parquet")
    silver["month"] = pd.to_datetime(silver["transaction_date"]).dt.to_period("M").dt.to_timestamp()
    med = silver.groupby(["area", "month"])["ppsf"].median()
    row = f.sample(1, random_state=0).iloc[0]
    prev = pd.Timestamp(row["month"]) - pd.DateOffset(months=1)
    assert row["area_ppsf_lag1"] == pytest.approx(med[(row["area"], prev)])
    assert f["area_ppsf_lag1"].notna().all() and f["area_ppsf_roll3"].notna().all()


def test_online_store_matches_offline_definition(built):
    import sqlite3
    with sqlite3.connect(config.ONLINE_DB) as cx:
        on = pd.read_sql("SELECT * FROM online_area_features", cx)
    assert set(on["area"]) == set(pd.read_parquet(config.GOLD_DIR / "area_month_stats.parquet")["area"])
    stats = pd.read_parquet(config.GOLD_DIR / "area_month_stats.parquet")
    a = on.iloc[0]["area"]
    g = stats[stats.area == a].sort_values("month")
    assert on.iloc[0]["area_ppsf_lag1"] == pytest.approx(g["median_ppsf"].iloc[-1])
    assert on.iloc[0]["area_ppsf_roll3"] == pytest.approx(g["median_ppsf"].tail(3).mean())
