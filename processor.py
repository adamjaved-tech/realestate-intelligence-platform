"""Near-real-time listing scorer.

A producer thread publishes listing events to a bounded queue (back-pressure); the
consumer micro-batches them, scores them with the production model and raises
*deal alerts* when an asking price deviates from the model's fair-value interval.

`EventSource` is the seam for Kafka / Pub/Sub: swap `SimulatedListingSource` for a
consumer that yields the same dict events and nothing else changes.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from propintel import config
from propintel.model.features import add_calendar, time_split
from propintel.model.predictor import REQUEST_COLS, Predictor

UNDERVALUED = 0.85  # ask / fair value below this -> undervalued; overpriced = ask > 1.10 x upper bound


class EventSource(Protocol):
    def __iter__(self) -> Iterator[dict]: ...


class SimulatedListingSource:
    """Replays held-out transactions as live listings with a noisy asking price."""

    def __init__(self, n_events: int = 1500, seed: int = 11, rate_per_s: float | None = None):
        df = add_calendar(pd.read_parquet(config.GOLD_DIR / "features.parquet"))
        _, _, test = time_split(df)
        # Replay only the latest month so event-time market state == online-store state.
        latest = test[test["month"] == test["month"].max()]
        self.events = latest.sample(min(n_events, len(latest)), random_state=seed)
        rng = np.random.default_rng(seed)
        mult = rng.lognormal(0, 0.06, len(self.events))
        flag = rng.random(len(self.events))
        mult = np.where(flag < 0.05, mult * 0.75, np.where(flag > 0.95, mult * 1.30, mult))  # planted deals
        self.events = self.events.assign(asking_price_aed=(self.events["price_aed"].values * mult).round(-3))
        self.rate = rate_per_s

    def __iter__(self) -> Iterator[dict]:
        for i, row in enumerate(self.events.to_dict("records")):
            if self.rate:
                time.sleep(1.0 / self.rate)
            ev = {c: row[c] for c in REQUEST_COLS}
            ev.update(listing_id=f"L{i:07d}", asking_price_aed=row["asking_price_aed"],
                      area=row["area"], emitted_at=time.time())
            yield ev


class ListingStreamProcessor:
    def __init__(self, predictor: Predictor, batch_size: int = 256, max_wait_s: float = 0.5,
                 alerts_path: Path | None = None, window: int = 500):
        self.p, self.batch_size, self.max_wait = predictor, batch_size, max_wait_s
        self.alerts_path = alerts_path or (config.MONITOR_DIR / "deal_alerts.jsonl")
        self.latencies: list[float] = []
        self.processed = 0
        self.alerts: list[dict] = []
        self.rolling = defaultdict(lambda: deque(maxlen=window))  # rolling ask $/sqft per area

    def _score(self, batch: list[dict]) -> None:
        df = self.p.predict_frame(pd.DataFrame(batch))
        now = time.time()
        for ev, row in zip(batch, df.to_dict("records"), strict=True):
            ratio = ev["asking_price_aed"] / row["predicted_price_aed"]
            self.rolling[ev["area"]].append(ev["asking_price_aed"] / ev["size_sqft"])
            self.latencies.append(now - ev["emitted_at"])
            kind = ("UNDERVALUED" if ratio < UNDERVALUED
                    else "OVERPRICED" if ev["asking_price_aed"] > row["upper_aed"] * 1.10 else None)
            if kind:
                self.alerts.append({
                    "listing_id": ev["listing_id"], "type": kind, "area": ev["area"],
                    "asking_price_aed": ev["asking_price_aed"], "fair_value_aed": row["predicted_price_aed"],
                    "fair_range_aed": [row["lower_aed"], row["upper_aed"]], "ask_to_fair": round(ratio, 3)})
        self.processed += len(batch)

    def run(self, source: EventSource, q_size: int = 2048) -> dict:
        q: queue.Queue = queue.Queue(maxsize=q_size)
        done = object()

        def produce() -> None:
            for ev in source:
                q.put(ev)  # blocks when consumer is slow -> back-pressure
            q.put(done)

        threading.Thread(target=produce, daemon=True).start()
        t0, buf, deadline, finished = time.time(), [], time.time() + self.max_wait, False
        while not finished:
            try:
                item = q.get(timeout=self.max_wait)
                if item is done:
                    finished = True
                else:
                    buf.append(item)
            except queue.Empty:
                pass
            if buf and (len(buf) >= self.batch_size or finished or time.time() >= deadline):
                self._score(buf)
                buf, deadline = [], time.time() + self.max_wait
        elapsed = time.time() - t0
        self._flush_alerts()
        lat = np.array(self.latencies)
        return {"events": self.processed, "alerts": len(self.alerts),
                "throughput_eps": round(self.processed / elapsed, 1),
                "event_to_score_p50_ms": round(float(np.percentile(lat, 50)) * 1000, 1),
                "event_to_score_p95_ms": round(float(np.percentile(lat, 95)) * 1000, 1)}

    def _flush_alerts(self) -> None:
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alerts_path.open("w") as f:
            for a in self.alerts:
                f.write(json.dumps(a) + "\n")


def main() -> None:
    p = Predictor()
    stats = ListingStreamProcessor(p).run(SimulatedListingSource())
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "stream_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
