"""Demonstrates drift detection: a stable sample vs. a shifted one (larger, pricier, villa-heavy listings)."""
import numpy as np
import pandas as pd

from propintel import config
from propintel.monitoring.drift import drift_report, write_report

ref = pd.read_parquet(config.MODELS_DIR / "reference.parquet")
rng = np.random.default_rng(0)
stable = ref.sample(2000, random_state=1)
shifted = ref.sample(2000, random_state=2).copy()
shifted["size_sqft"] *= rng.lognormal(0.25, 0.1, len(shifted))
shifted["pred_to_baseline_ratio"] *= 1.18
shifted.loc[shifted.sample(frac=0.3, random_state=3).index, "area"] = "Palm Jumeirah"

for name, cur in [("stable", stable), ("shifted", shifted)]:
    rep = drift_report(ref, cur)
    print(f"{name:8s} -> {rep['status']:6s} drifted={rep['drifted_features']}")
    if name == "shifted":
        print(f"report written to {write_report(rep)}")
