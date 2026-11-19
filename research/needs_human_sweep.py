"""Sweep THREAT_NEEDS_HUMAN_THRESHOLD on Civil subsample."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput


def sweep_needs_human(
    texts: list[str],
    ids: list[str],
    thresholds: list[float] | None = None,
    *,
    use_transformer: bool = False,
) -> list[dict[str, Any]]:
    thresholds = thresholds or [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows: list[dict[str, Any]] = []
    n = len(texts)
    comments = [CommentInput(id=i, text=t) for i, t in zip(ids, texts)]
    for thr in thresholds:
        os.environ["THREAT_NEEDS_HUMAN_THRESHOLD"] = str(thr)
        pipe = ModerationPipeline(use_transformer=use_transformer)
        resp = pipe.analyze_batch(comments)
        nh = sum(1 for r in resp.results if r.needs_human)
        rows.append(
            {
                "threat_needs_human_threshold": thr,
                "needs_human_count": nh,
                "needs_human_rate": nh / max(n, 1),
                "use_transformer": use_transformer,
            }
        )
    return rows


def scenario_needs_human_rate(
    texts: list[str],
    ids: list[str],
    *,
    use_transformer: bool,
) -> dict[str, Any]:
    pipe = ModerationPipeline(use_transformer=use_transformer)
    resp = pipe.analyze_batch([CommentInput(id=i, text=t) for i, t in zip(ids, texts)])
    nh = sum(1 for r in resp.results if r.needs_human)
    return {
        "needs_human_count": nh,
        "needs_human_rate": nh / max(len(texts), 1),
        "n": len(texts),
    }