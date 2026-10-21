"""Grounded market-intelligence assistant (RAG).

Knowledge base = (a) quarterly market commentary generated from the gold layer and
(b) hand-written methodology docs in docs/knowledge. Retrieval is TF-IDF + metadata
filters (area / period) so answers are deterministic, cheap and citable. Generation is
extractive by default; set ANTHROPIC_API_KEY to let an LLM synthesise the final answer
from the SAME retrieved passages (it is instructed to cite and to refuse if unsupported).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from propintel import config
from propintel.domain import AREAS


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    area: str | None = None
    period: str | None = None  # e.g. "2025-Q3"


def build_market_chunks(stats: pd.DataFrame) -> list[Chunk]:
    s = stats.copy()
    s["month"] = pd.to_datetime(s["month"])
    s["period"] = s["month"].dt.year.astype(str) + "-Q" + s["month"].dt.quarter.astype(str)
    q = (s.groupby(["area", "period"])
          .agg(ppsf=("median_ppsf", "mean"), txns=("txn_count", "sum"), last=("month", "max"))
          .reset_index().sort_values(["area", "last"]))
    q["qoq"] = q.groupby("area")["ppsf"].pct_change()
    q["yoy"] = q.groupby("area")["ppsf"].pct_change(4)
    chunks = []
    for r in q.itertuples():
        qoq = f"{r.qoq:+.1%} versus the previous quarter" if pd.notna(r.qoq) else "no prior quarter available"
        yoy = f"{r.yoy:+.1%} year over year" if pd.notna(r.yoy) else "no year-over-year comparison available"
        text = (f"{r.area} market report {r.period}: the median price per sqft averaged AED {r.ppsf:,.0f}, "
                f"{qoq} and {yoy}. {int(r.txns):,} transactions were recorded.")
        chunks.append(Chunk(f"market:{r.area}:{r.period}", text, "gold/area_month_stats", r.area, r.period))
    # city-level quarter summary + ranking
    cq = q.groupby("period").agg(ppsf=("ppsf", "mean"), txns=("txns", "sum")).reset_index()
    for p in cq.itertuples():
        top = q[q.period == p.period].sort_values("ppsf", ascending=False).iloc[0]
        chunks.append(Chunk(
            f"market:city:{p.period}",
            f"Dubai citywide market summary {p.period}: average median price per sqft across areas AED {p.ppsf:,.0f} "
            f"with {int(p.txns):,} transactions. The most expensive area was {top.area} at AED {top.ppsf:,.0f} per sqft.",
            "gold/area_month_stats", None, p.period))
    return chunks


def load_knowledge_chunks(folder: Path | None = None) -> list[Chunk]:
    folder = folder or config.KNOWLEDGE_DIR
    out = []
    for f in sorted(folder.glob("*.md")):
        for i, para in enumerate(p.strip() for p in f.read_text().split("\n\n") if len(p.strip()) > 40):
            out.append(Chunk(f"doc:{f.stem}:{i}", para, f"docs/knowledge/{f.name}"))
    return out


_PERIOD = re.compile(r"q([1-4])\s*[-/ ]?\s*(20\d\d)|(20\d\d)\s*[-/ ]?\s*q([1-4])", re.IGNORECASE)


def parse_period(query: str) -> str | None:
    m = _PERIOD.search(query)
    if not m:
        return None
    q, y = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
    return f"{y}-Q{q}"


def parse_area(query: str) -> str | None:
    ql = query.lower()
    return next((a for a in AREAS if a.lower() in ql), None)


@dataclass
class Answer:
    answer: str
    citations: list[dict] = field(default_factory=list)
    mode: str = "extractive"


class MarketAssistant:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([c.text for c in chunks])

    @classmethod
    def from_gold(cls) -> MarketAssistant:
        stats = pd.read_parquet(config.GOLD_DIR / "area_month_stats.parquet")
        return cls(build_market_chunks(stats) + load_knowledge_chunks())

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        area, period = parse_area(query), parse_period(query)
        scores = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        for i, c in enumerate(self.chunks):  # metadata filters become hard constraints when present
            is_market = c.source.startswith("gold")
            if area and is_market and c.area != area:
                scores[i] = -1
            if period and c.period is not None and c.period != period:
                scores[i] = -1
        idx = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in idx if scores[i] > 0]

    def ask(self, query: str, k: int = 3) -> Answer:
        hits = self.retrieve(query, k)
        if not hits:
            return Answer("I could not find supporting information in the knowledge base for that question.", [], "refusal")
        cites = [{"ref": i + 1, "id": c.id, "source": c.source, "score": round(s, 3)} for i, (c, s) in enumerate(hits)]
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return Answer(self._llm_answer(query, hits), cites, "llm")
            except Exception:  # never fail the request because the LLM is unavailable
                pass
        top = hits[0][1]
        keep = [(i, c) for i, (c, sc) in enumerate(hits) if sc >= max(0.05, 0.25 * top)]
        body = " ".join(f"{c.text} [{i + 1}]" for i, c in keep)
        return Answer(body, cites, "extractive")

    def _llm_answer(self, query: str, hits: list[tuple[Chunk, float]]) -> str:  # pragma: no cover
        import anthropic
        ctx = "\n".join(f"[{i + 1}] {c.text}" for i, (c, _) in enumerate(hits))
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("PROPINTEL_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=400,
            system="Answer ONLY from the numbered context. Cite sources like [1]. If the context does not "
                   "contain the answer, say you cannot answer.",
            messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}])
        return msg.content[0].text
