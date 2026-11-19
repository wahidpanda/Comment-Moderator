"""Shared evaluation statistics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score


def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(float(f1_score(y_true[idx], y_pred[idx], zero_division=0)))
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_ci95_low": float(np.percentile(scores, 2.5)),
        "f1_ci95_high": float(np.percentile(scores, 97.5)),
        "n_bootstrap": n_boot,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }