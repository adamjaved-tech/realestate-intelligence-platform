"""FastAPI service: valuation, batch valuation, market assistant, drift, metrics."""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from propintel import __version__, config
from propintel.domain import AREAS
from propintel.model.predictor import Predictor
from propintel.monitoring.drift import drift_report, load_logged_requests
from propintel.rag.assistant import MarketAssistant

REQS = Counter("propintel_requests_total", "HTTP requests", ["path", "status"])
LAT = Histogram("propintel_request_seconds", "Request latency", ["path"],
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDS = Counter("propintel_predictions_total", "Valuations served", ["area"])
_LOG_LOCK = threading.Lock()

Area = Literal[tuple(AREAS)]  # type: ignore[valid-type]


class Property(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "area": "Dubai Marina", "property_type": "Apartment", "bedrooms": 2, "bathrooms": 3,
        "size_sqft": 1250, "floor": 22, "age_years": 4, "view": "Sea", "furnished": 0, "parking_spots": 1}})
    area: Area
    property_type: Literal["Apartment", "Townhouse", "Villa"]
    bedrooms: int = Field(ge=0, le=8)
    bathrooms: int = Field(ge=1, le=10)
    size_sqft: float = Field(gt=100, lt=30000)
    floor: int = Field(ge=0, le=120, default=0)
    age_years: int = Field(ge=0, le=60, default=0)
    view: Literal["Sea", "City", "Community", "None"] = "None"
    furnished: Literal[0, 1] = 0
    parking_spots: int = Field(ge=0, le=10, default=1)


class Valuation(BaseModel):
    predicted_price_aed: float
    lower_aed: float
    upper_aed: float
    predicted_ppsf: float
    model_version: str
    interval: str = "80% prediction interval"


class BatchRequest(BaseModel):
    properties: list[Property] = Field(min_length=1, max_length=1000)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _log_predictions(df: pd.DataFrame) -> None:
    config.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["area", "property_type", "bedrooms", "bathrooms", "size_sqft", "floor", "age_years", "view",
            "furnished", "parking_spots", "predicted_price_aed", "pred_to_baseline_ratio"]
    with _LOG_LOCK, (config.MONITOR_DIR / "predictions.jsonl").open("a") as f:
        for r in df[cols].to_dict("records"):
            f.write(json.dumps({"ts": time.time(), **r}) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    try:
        app.state.assistant = MarketAssistant.from_gold()
    except FileNotFoundError:
        app.state.assistant = None
    app.state.reference = pd.read_parquet(config.MODELS_DIR / "reference.parquet")
    yield


app = FastAPI(title="PropIntel API", version=__version__,
              description="Property valuation with prediction intervals + market intelligence assistant.",
              lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    REQS.labels(path, response.status_code).inc()
    LAT.labels(path).observe(dt)
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = f"{dt * 1000:.1f}"
    return response


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready", tags=["ops"])
def ready(request: Request):
    return {"ready": request.app.state.predictor is not None, "model_version": request.app.state.predictor.version}


@app.get("/metrics", tags=["ops"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model", tags=["model"])
def model_info(request: Request):
    b = request.app.state.predictor.bundle
    return {"version": b["version"], "trained_at": b["trained_at"], "train_window": b["train_window"],
            "test_window": b["test_window"], "metrics": b["metrics"], "features": b["features"]}


@app.get("/v1/areas", tags=["model"])
def areas(request: Request):
    return request.app.state.predictor.online.reset_index().to_dict("records")


def _value(request: Request, props: list[Property]) -> pd.DataFrame:
    try:
        df = request.app.state.predictor.predict_frame(pd.DataFrame([p.model_dump() for p in props]))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    for a in df["area"]:
        PREDS.labels(a).inc()
    _log_predictions(df)
    return df


def _to_valuation(df: pd.DataFrame, version: str) -> list[Valuation]:
    return [Valuation(model_version=version, **r) for r in
            df[["predicted_price_aed", "lower_aed", "upper_aed", "predicted_ppsf"]].to_dict("records")]


@app.post("/v1/valuations", response_model=Valuation, tags=["valuation"])
def valuation(prop: Property, request: Request):
    df = _value(request, [prop])
    return _to_valuation(df, request.app.state.predictor.version)[0]


@app.post("/v1/valuations/batch", response_model=list[Valuation], tags=["valuation"])
def valuation_batch(body: BatchRequest, request: Request):
    df = _value(request, body.properties)
    return _to_valuation(df, request.app.state.predictor.version)


@app.post("/v1/assistant/ask", tags=["assistant"])
def ask(q: Question, request: Request):
    if request.app.state.assistant is None:
        raise HTTPException(503, "knowledge base not built; run the pipeline first")
    a = request.app.state.assistant.ask(q.question)
    return {"answer": a.answer, "citations": a.citations, "mode": a.mode}


@app.get("/v1/monitoring/drift", tags=["monitoring"])
def drift(request: Request, min_rows: int = 200):
    cur = load_logged_requests()
    if cur.empty:
        return {"status": "INSUFFICIENT_DATA", "rows": 0, "features": {}}
    return drift_report(request.app.state.reference, cur, min_rows=min_rows)
