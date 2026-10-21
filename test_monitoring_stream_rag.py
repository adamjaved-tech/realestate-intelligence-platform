import numpy as np
import pandas as pd

from propintel import config
from propintel.domain import AREAS
from propintel.model.predictor import Predictor
from propintel.monitoring.drift import drift_report, performance_report, psi_categorical, psi_numeric
from propintel.rag.assistant import MarketAssistant, parse_area, parse_period
from propintel.streaming.processor import ListingStreamProcessor, SimulatedListingSource


def test_psi_identical_is_zero_and_shift_is_detected():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi_numeric(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi_numeric(a, rng.normal(1.5, 1, 5000)) > 0.25
    s = pd.Series(["a"] * 80 + ["b"] * 20)
    assert psi_categorical(s, s) == 0.0
    assert psi_categorical(s, pd.Series(["a"] * 20 + ["b"] * 80)) > 0.25


def test_drift_report_stable_vs_shifted(built):
    ref = pd.read_parquet(config.MODELS_DIR / "reference.parquet")
    assert drift_report(ref, ref.sample(1500, random_state=1))["status"] == "OK"
    shifted = ref.sample(1500, random_state=2).copy()
    shifted["size_sqft"] *= 1.8
    rep = drift_report(ref, shifted)
    assert rep["status"] == "ALERT" and "size_sqft" in rep["drifted_features"]
    assert drift_report(ref, ref.head(5))["status"] == "INSUFFICIENT_DATA"


def test_performance_alert():
    y = np.array([100.0, 200, 300])
    assert performance_report(y, y * 1.05, 0.07)["status"] == "OK"
    assert performance_report(y, y * 1.30, 0.07)["status"] == "ALERT"


def test_stream_scores_everything_and_flags_planted_deals(built, tmp_path):
    proc = ListingStreamProcessor(Predictor(), batch_size=64, alerts_path=tmp_path / "a.jsonl")
    stats = proc.run(SimulatedListingSource(n_events=400))
    assert stats["events"] > 0 and stats["throughput_eps"] > 0
    assert 0 < stats["alerts"] < stats["events"] * 0.4
    assert {a["type"] for a in proc.alerts} <= {"UNDERVALUED", "OVERPRICED"}
    assert (tmp_path / "a.jsonl").exists()


def test_rag_retrieval_hits_correct_area_and_period(built):
    bot = MarketAssistant.from_gold()
    for area in AREAS:
        hits = bot.retrieve(f"What was the median price per sqft in {area} in Q2 2024?", k=1)
        assert hits[0][0].id == f"market:{area}:2024-Q2"


def test_rag_refuses_when_unsupported(built):
    bot = MarketAssistant.from_gold()
    assert bot.ask("zzzz qqqq xxxx").mode == "refusal"


def test_query_parsers():
    assert parse_period("results for 2025 Q3") == "2025-Q3" and parse_period("Q1-2023") == "2023-Q1"
    assert parse_area("how is palm jumeirah doing") == "Palm Jumeirah"
