"""Threat detection recall helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.schemas import ModerationLabel


def predict_threat(result: Any, *, score_threshold: float = 0.5) -> bool:
    if result.label == ModerationLabel.THREAT:
        return True
    return float(result.scores.threat) >= score_threshold


def threat_recall(rows: list[tuple[np.ndarray | None, list[Any]]]) -> dict[str, float | int]:
    """rows: (y_threat, results) pairs combined."""
    all_true: list[int] = []
    all_pred: list[int] = []
    for y_true, results in rows:
        if y_true is None:
            continue
        preds = [1 if predict_threat(r) else 0 for r in results]
        all_true.extend(int(x) for x in y_true)
        all_pred.extend(preds)
    if not all_true:
        return {"n": 0, "recall": None, "true_threats": 0, "detected": 0}
    yt = np.asarray(all_true, dtype=np.int8)
    yp = np.asarray(all_pred, dtype=np.int8)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    recall = tp / max(tp + fn, 1)
    return {
        "n": len(yt),
        "true_threats": int(yt.sum()),
        "detected": tp,
        "recall": float(recall),
    }


def civil_threat_mask(test_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    if "threat" not in test_df.columns:
        return np.zeros(len(test_df), dtype=np.int8)
    return (test_df["threat"].astype(float).to_numpy() >= threshold).astype(np.int8)