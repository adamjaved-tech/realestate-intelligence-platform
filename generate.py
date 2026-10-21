"""Generate reproducible synthetic Dubai property transactions for demos and tests.

No real listings, transactions, customer records, or employer data are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from propintel import config
from propintel.domain import AREAS, PROPERTY_TYPES, START_MONTH, VIEWS


def generate_clean(n: int = 60_000, seed: int = 42) -> pd.DataFrame:
    """Create realistic-ish synthetic transactions with stable, reproducible rows."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    area_names = list(AREAS)
    weights = np.array([AREAS[a]["w"] for a in area_names], dtype=float)
    weights /= weights.sum()
    area = rng.choice(area_names, size=n, p=weights)
    month_idx = rng.integers(0, 48, size=n)
    dates = pd.to_datetime([(pd.Period(START_MONTH, freq="M") + int(i)).to_timestamp() for i in month_idx])
    prop_types = []
    for a in area:
        p_villa = AREAS[a]["villa"]
        r = rng.random()
        prop_types.append("Villa" if r < p_villa else ("Townhouse" if r < p_villa + 0.16 else "Apartment"))
    property_type = np.array(prop_types)
    bedrooms = np.where(property_type == "Apartment", rng.integers(0, 5, n), rng.integers(2, 7, n))
    size = np.where(property_type == "Apartment", rng.normal(850 + bedrooms * 230, 180, n),
                    rng.normal(1700 + bedrooms * 420, 420, n))
    size = np.clip(size, 300, 8000).round(0)
    floor = np.array([rng.integers(1, max(2, AREAS[a]["tall"] + 1)) for a in area])
    age = rng.integers(0, 31, n)
    furnished = rng.binomial(1, 0.28, n)
    parking = np.clip((bedrooms / 2).astype(int) + rng.binomial(1, .25, n), 0, 4)
    views = []
    for a in area:
        p_sea = AREAS[a]["sea"]
        views.append("Sea" if rng.random() < p_sea else rng.choice(["City", "Community", "None"], p=[.43, .37, .20]))
    base = np.array([AREAS[a]["ppsf"] for a in area], dtype=float)
    growth = np.array([AREAS[a]["growth"] for a in area], dtype=float)
    time_factor = np.exp(growth * month_idx / 12)
    type_factor = np.where(property_type == "Villa", 1.10, np.where(property_type == "Townhouse", 1.04, 1.0))
    view_factor = np.where(np.array(views) == "Sea", 1.16, np.where(np.array(views) == "City", 1.035, 1.0))
    floor_factor = 1 + np.clip(floor, 0, 50) * .0012
    age_factor = np.maximum(.78, 1 - age * .004)
    seasonal = 1 + .012 * np.sin(2 * np.pi * (month_idx % 12) / 12)
    noise = rng.lognormal(mean=-.5 * .08**2, sigma=.08, size=n)
    ppsf = base * time_factor * type_factor * view_factor * floor_factor * age_factor * seasonal * noise
    price = (ppsf * size).round(-2)
    return pd.DataFrame({
        "transaction_id": [f"TX{seed:04d}{i:07d}" for i in range(n)],
        "transaction_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "area": area,
        "property_type": property_type,
        "bedrooms": bedrooms.astype(int),
        "bathrooms": np.clip(bedrooms + rng.integers(0, 3, n), 1, 8),
        "size_sqft": size,
        "floor": floor.astype(int),
        "age_years": age.astype(int),
        "view": np.array(views),
        "furnished": furnished.astype(int),
        "parking_spots": parking.astype(int),
        "price_aed": price.astype(float),
    })


def inject_dirt(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject controlled data-quality issues (duplicates, nulls, malformed values/outliers)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    duplicate_count = max(1, round(len(out) * .01))
    out = pd.concat([out, out.sample(n=duplicate_count, random_state=seed)], ignore_index=True)
    n = len(out)
    # Deterministic, non-overlapping-ish dirty row groups for pipeline demonstrations.
    idx = rng.choice(n, size=max(1, round(n * .012)), replace=False)
    for i, row in enumerate(idx):
        mode = i % 5
        if mode == 0:
            out.loc[row, "size_sqft"] = np.nan
        elif mode == 1:
            out.loc[row, "transaction_date"] = "not-a-date"
        elif mode == 2:
            out.loc[row, "price_aed"] = -1
        elif mode == 3:
            out.loc[row, "area"] = "Unknown District"
        else:
            out.loc[row, "price_aed"] = 50_000_000
    # Small casing/whitespace noise is normalised by the silver layer.
    noise_idx = rng.choice(n, size=max(1, round(n * .01)), replace=False)
    out.loc[noise_idx, "area"] = out.loc[noise_idx, "area"].astype(str).str.upper()
    return out.reset_index(drop=True)


def generate_raw(n: int = 60_000, seed: int = 42, output: Path | None = None) -> Path:
    """Write the dirty synthetic CSV consumed by the medallion pipeline."""
    output = output or (config.RAW_DIR / "transactions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_dirty = inject_dirt(generate_clean(n=n, seed=seed), seed=seed)
    generate_dirty.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=60_000, help="number of clean synthetic transactions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = generate_raw(n=args.rows, seed=args.seed)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
