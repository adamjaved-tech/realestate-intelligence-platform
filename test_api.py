import pytest
from fastapi.testclient import TestClient

from tests.test_model import PROP


@pytest.fixture(scope="module")
def client(built):
    from propintel.serving.app import app
    with TestClient(app) as c:
        yield c


def test_health_ready_model(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True
    info = client.get("/v1/model").json()
    assert info["metrics"]["test_mape"] < info["metrics"]["baseline_mape"]


def test_single_valuation(client):
    r = client.post("/v1/valuations", json=PROP)
    assert r.status_code == 200
    b = r.json()
    assert b["lower_aed"] <= b["predicted_price_aed"] <= b["upper_aed"]
    assert "x-request-id" in r.headers


@pytest.mark.parametrize("patch", [{"area": "Mars"}, {"bedrooms": -1}, {"size_sqft": 5}, {"view": "Moon"}])
def test_validation_rejects_bad_input(client, patch):
    assert client.post("/v1/valuations", json={**PROP, **patch}).status_code == 422


def test_batch_and_metrics(client):
    r = client.post("/v1/valuations/batch", json={"properties": [PROP] * 50})
    assert r.status_code == 200 and len(r.json()) == 50
    assert client.post("/v1/valuations/batch", json={"properties": []}).status_code == 422
    m = client.get("/metrics").text
    assert "propintel_predictions_total" in m and "propintel_request_seconds_bucket" in m


def test_assistant_is_grounded(client):
    r = client.post("/v1/assistant/ask", json={"question": "Median price per sqft in Dubai Marina in Q3 2025?"}).json()
    assert r["citations"][0]["id"] == "market:Dubai Marina:2025-Q3"
    assert "AED" in r["answer"]


def test_drift_endpoint(client):
    assert client.get("/v1/monitoring/drift", params={"min_rows": 10}).json()["status"] in {"OK", "WARN", "ALERT"}
