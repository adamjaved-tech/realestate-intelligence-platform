"""Builds a small, isolated end-to-end environment once per test session."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="propintel-test-")
os.environ["PROPINTEL_HOME"] = _TMP
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP}/mlflow.db"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built():
    from propintel import config
    from propintel.data.generate import generate_raw
    from propintel.model.train import train
    from propintel.pipeline.medallion import run_all

    shutil.copytree(REPO / "docs" / "knowledge", config.KNOWLEDGE_DIR)
    generate_raw(n=15_000, seed=1)
    result = {"pipeline": run_all()}
    result["train"] = train(n_estimators=150, register=True)
    return result
