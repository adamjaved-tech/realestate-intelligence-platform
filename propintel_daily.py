"""Airflow DAG: daily ETL -> quality gate -> retrain -> drift check -> refresh online features.

Deploy by mounting this folder into $AIRFLOW_HOME/dags with `propintel` installed in the
worker image. Each task wraps the same importable functions the CLI and tests use.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

default_args = {"owner": "data-platform", "retries": 2, "retry_delay": timedelta(minutes=5),
                "retry_exponential_backoff": True, "sla": timedelta(hours=1)}


@dag(dag_id="propintel_daily", schedule="0 3 * * *", start_date=datetime(2025, 1, 1),
     catchup=False, default_args=default_args, tags=["propintel", "ml"], max_active_runs=1)
def propintel_daily():
    @task
    def bronze() -> str:
        from propintel.pipeline.medallion import run_bronze
        return str(run_bronze())

    @task
    def silver(_: str) -> dict:
        from propintel.pipeline.medallion import run_silver
        return run_silver()  # raises DataQualityError -> task fails -> downstream blocked

    @task
    def gold(_: dict) -> dict:
        from propintel.pipeline.medallion import run_gold
        return run_gold()

    @task
    def retrain(_: dict) -> dict:
        from propintel.model.train import train
        return train()

    @task
    def drift_check(_: dict) -> str:
        import pandas as pd
        from propintel import config
        from propintel.monitoring.drift import drift_report, load_logged_requests, write_report
        rep = drift_report(pd.read_parquet(config.MODELS_DIR / "reference.parquet"), load_logged_requests())
        write_report(rep)
        if rep["status"] == "ALERT":
            raise RuntimeError(f"drift alert on {rep['drifted_features']}")
        return rep["status"]

    drift_check(retrain(gold(silver(bronze()))))


propintel_daily()
