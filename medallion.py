"""Batch ETL: raw CSV -> bronze -> silver (validated) -> gold (features + market stats).

DuckDB does the heavy lifting in SQL; every layer is Parquet so the same code maps
1:1 onto BigQuery/Spark/S3 in a cloud deployment.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

from propintel import config
from propintel.domain import AREAS, PROPERTY_TYPES

log = logging.getLogger("propintel.pipeline")


class DataQualityError(RuntimeError):
    """Raised when the silver layer rejects more rows than the configured gate."""


# --------------------------------------------------------------------------- bronze
def run_bronze(raw: Path | None = None) -> Path:
    raw = raw or (config.RAW_DIR / "transactions.csv")
    config.BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.BRONZE_DIR / "transactions.parquet"
    con = duckdb.connect()
    # all_varchar: bronze never fails on bad values; typing happens in silver.
    con.execute(f"""
        COPY (SELECT *, current_timestamp AS _ingested_at, '{raw.name}' AS _source_file
              FROM read_csv('{raw}', header = true, all_varchar = true))
        TO '{out}' (FORMAT parquet)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    log.info("bronze: %s rows", n)
    return out


# --------------------------------------------------------------------------- silver
def run_silver() -> dict:
    config.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    bronze = config.BRONZE_DIR / "transactions.parquet"
    con = duckdb.connect()

    area_map = pd.DataFrame({"k": [a.lower() for a in AREAS], "area": list(AREAS)})
    type_map = pd.DataFrame({"k": [t.lower() for t in PROPERTY_TYPES], "property_type": PROPERTY_TYPES})
    con.register("area_map", area_map)
    con.register("type_map", type_map)

    con.execute(f"""
        CREATE TABLE deduped AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY _ingested_at) AS rn
            FROM read_parquet('{bronze}')) WHERE rn = 1""")

    con.execute("""
        CREATE TABLE typed AS
        SELECT d.transaction_id,
               try_cast(d.transaction_date AS DATE)       AS transaction_date,
               am.area                                      AS area,
               tm.property_type                             AS property_type,
               try_cast(d.bedrooms AS INTEGER)              AS bedrooms,
               try_cast(d.bathrooms AS INTEGER)             AS bathrooms,
               try_cast(d.size_sqft AS DOUBLE)              AS size_sqft,
               try_cast(d.floor AS INTEGER)                 AS floor,
               try_cast(d.age_years AS INTEGER)             AS age_years,
               d.view                                       AS view,
               try_cast(d.furnished AS INTEGER)             AS furnished,
               try_cast(d.parking_spots AS INTEGER)         AS parking_spots,
               try_cast(d.price_aed AS DOUBLE)              AS price_aed
        FROM deduped d
        LEFT JOIN area_map am ON lower(trim(d.area)) = am.k
        LEFT JOIN type_map tm ON lower(trim(d.property_type)) = tm.k""")

    con.execute("""
        CREATE TABLE checked AS
        SELECT *, price_aed / NULLIF(size_sqft, 0) AS ppsf,
          CASE
            WHEN transaction_date IS NULL                         THEN 'invalid_date'
            WHEN area IS NULL                                     THEN 'unknown_area'
            WHEN property_type IS NULL                            THEN 'unknown_property_type'
            WHEN price_aed IS NULL OR price_aed <= 0              THEN 'invalid_price'
            WHEN size_sqft IS NULL OR size_sqft <= 0              THEN 'missing_size'
            WHEN bedrooms IS NULL OR bedrooms NOT BETWEEN 0 AND 8 THEN 'invalid_bedrooms'
            WHEN price_aed / size_sqft NOT BETWEEN 200 AND 15000  THEN 'price_outlier'
            ELSE NULL END AS reject_reason
        FROM typed""")

    valid = config.SILVER_DIR / "transactions.parquet"
    quarantine = config.SILVER_DIR / "quarantine.parquet"
    con.execute(f"COPY (SELECT * EXCLUDE (reject_reason) FROM checked WHERE reject_reason IS NULL) TO '{valid}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM checked WHERE reject_reason IS NOT NULL) TO '{quarantine}' (FORMAT parquet)")

    rows_in = con.execute(f"SELECT count(*) FROM read_parquet('{bronze}')").fetchone()[0]
    rows_dedup = con.execute("SELECT count(*) FROM deduped").fetchone()[0]
    reasons = dict(con.execute(
        "SELECT reject_reason, count(*) FROM checked WHERE reject_reason IS NOT NULL GROUP BY 1").fetchall())
    n_valid = con.execute("SELECT count(*) FROM checked WHERE reject_reason IS NULL").fetchone()[0]
    report = {
        "rows_in": rows_in,
        "duplicates_removed": rows_in - rows_dedup,
        "rows_valid": n_valid,
        "rows_quarantined": sum(reasons.values()),
        "quarantine_by_reason": reasons,
        "reject_rate": round(sum(reasons.values()) / max(rows_dedup, 1), 5),
        "gate": config.MAX_REJECT_RATE,
    }
    report["passed"] = report["reject_rate"] <= config.MAX_REJECT_RATE
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    log.info("silver: %s", report)
    if not report["passed"]:
        raise DataQualityError(f"reject rate {report['reject_rate']:.2%} exceeds gate {config.MAX_REJECT_RATE:.0%}")
    return report


# --------------------------------------------------------------------------- gold
_STATS_SQL = """
WITH m AS (
    SELECT area, date_trunc('month', transaction_date)::DATE AS month,
           median(ppsf) AS median_ppsf, count(*) AS txn_count
    FROM read_parquet('{silver}') GROUP BY ALL),
c AS (
    SELECT date_trunc('month', transaction_date)::DATE AS month, median(ppsf) AS city_ppsf
    FROM read_parquet('{silver}') GROUP BY ALL)
SELECT m.area, m.month, m.median_ppsf, m.txn_count, c.city_ppsf,
       row_number() OVER w AS rn,
       lag(m.median_ppsf, 1) OVER w AS area_ppsf_lag1,
       avg(m.median_ppsf) OVER (PARTITION BY m.area ORDER BY m.month
                                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS area_ppsf_roll3,
       lag(m.txn_count, 1) OVER w AS area_txn_lag1,
       lag(m.median_ppsf, 1) OVER w / NULLIF(lag(m.median_ppsf, 2) OVER w, 0) - 1 AS area_mom_growth,
       lag(c.city_ppsf, 1) OVER w AS city_ppsf_lag1
FROM m JOIN c USING (month)
WINDOW w AS (PARTITION BY m.area ORDER BY m.month)
ORDER BY m.area, m.month"""


def run_gold() -> dict:
    """Market stats + leakage-safe training table.

    Every market feature for a transaction in month M is computed ONLY from months < M
    (lag1 / rolling mean of the 3 prior months), so there is no target leakage.
    """
    config.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    silver = config.SILVER_DIR / "transactions.parquet"
    con = duckdb.connect()
    stats_path = config.GOLD_DIR / "area_month_stats.parquet"
    con.execute(f"COPY ({_STATS_SQL.format(silver=silver)}) TO '{stats_path}' (FORMAT parquet)")

    feats = config.GOLD_DIR / "features.parquet"
    con.execute(f"""
        COPY (
          SELECT t.transaction_id, t.transaction_date,
                 date_trunc('month', t.transaction_date)::DATE AS month,
                 t.area, t.property_type, t.bedrooms, t.bathrooms, t.size_sqft, t.floor,
                 t.age_years, t.view, t.furnished, t.parking_spots, t.price_aed, t.ppsf,
                 s.area_ppsf_lag1, s.area_ppsf_roll3, s.area_txn_lag1, s.area_mom_growth,
                 s.city_ppsf_lag1
          FROM read_parquet('{silver}') t
          JOIN read_parquet('{stats_path}') s
            ON s.area = t.area AND s.month = date_trunc('month', t.transaction_date)::DATE
          WHERE s.rn > 3 AND s.area_mom_growth IS NOT NULL
        ) TO '{feats}' (FORMAT parquet)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{feats}')").fetchone()[0]
    materialize_online_features()
    log.info("gold: %s feature rows", n)
    return {"feature_rows": n}


def materialize_online_features() -> pd.DataFrame:
    """Push the latest market state per area into the low-latency online store (SQLite).

    Equivalent to a feature-store materialisation job: the row is what the offline
    pipeline WOULD compute for a transaction in the month following the data.
    """
    stats = pd.read_parquet(config.GOLD_DIR / "area_month_stats.parquet").sort_values(["area", "month"])
    rows = []
    for area, g in stats.groupby("area"):
        last3 = g.tail(3)
        rows.append({
            "area": area,
            "as_of_month": (pd.Timestamp(g["month"].iloc[-1]) + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
            "area_ppsf_lag1": float(last3["median_ppsf"].iloc[-1]),
            "area_ppsf_roll3": float(last3["median_ppsf"].mean()),
            "area_txn_lag1": float(last3["txn_count"].iloc[-1]),
            "area_mom_growth": float(last3["median_ppsf"].iloc[-1] / last3["median_ppsf"].iloc[-2] - 1),
            "city_ppsf_lag1": float(last3["city_ppsf"].iloc[-1]),
        })
    df = pd.DataFrame(rows)
    config.ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.ONLINE_DB) as cx:
        df.to_sql("online_area_features", cx, if_exists="replace", index=False)
    return df


def run_all(raw: Path | None = None) -> dict:
    run_bronze(raw)
    dq = run_silver()
    gold = run_gold()
    return {"data_quality": dq, **gold}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(run_all(), indent=2))


if __name__ == "__main__":
    main()
