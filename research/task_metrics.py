"""Decomposed toxicity / spam / threat metrics (spam never counts as toxic FP)."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.schemas import ModerationLabel
from research.eval import HARMFUL_LABELS
from research.stats import bootstrap_f1_ci, wilson_ci
from research.threat_metrics import predict_threat


def _is_spam_decision(r: Any) -> bool:
    return r.decided_by == "spam" or r.label == ModerationLabel.SPAM


def pred_toxicity_harmful(r: Any) -> int:
    if _is_spam_decision(r):
        return 0
    if r.label == ModerationLabel.THREAT:
        return 1
    if r.label == ModerationLabel.IDENTITY_ATTACK:
        return 1
    if r.label == ModerationLabel.TOXIC:
        return 1
    return 1 if r.label.value in HARMFUL_LABELS else 0


def pred_spam(r: Any) -> int:
    return 1 if _is_spam_decision(r) else 0


def pred_threat_bin(r: Any) -> int:
    return 1 if predict_threat(r) else 0


def task_metrics(
    y_toxic: np.ndarray,
    y_spam: np.ndarray | None,
    y_threat: np.ndarray,
    results: list[Any],
    *,
    task_name: str,
) -> dict[str, Any]:
    y_tox_p = np.array([pred_toxicity_harmful(r) for r in results], dtype=np.int8)
    y_spam_p = np.array([pred_spam(r) for r in results], dtype=np.int8)
    y_thr_p = np.array([pred_threat_bin(r) for r in results], dtype=np.int8)

    out: dict[str, Any] = {"task_eval": task_name}
    benign = y_toxic == 0
    tox_fp = int(((y_tox_p == 1) & benign).sum())
    n_benign = int(benign.sum())
    lo, hi = wilson_ci(tox_fp, n_benign)
    f1 = bootstrap_f1_ci(y_toxic, y_tox_p, n_boot=500)
    out["toxicity"] = {
        **f1,
        "false_positives_benign": tox_fp,
        "benign_count": n_benign,
        "benign_fpr": tox_fp / max(n_benign, 1),
        "benign_fpr_ci95_low": lo,
        "benign_fpr_ci95_high": hi,
        "note": "Spam-layer decisions excluded from toxic FP",
    }

    if y_spam is not None:
        f1s = bootstrap_f1_ci(y_spam, y_spam_p, n_boot=500)
        out["spam"] = {**f1s, "n_spam_decisions": int(y_spam_p.sum())}

    if y_threat.any():
        f1t = bootstrap_f1_ci(y_threat, y_thr_p, n_boot=200)
        out["threat"] = {
            **f1t,
            "positives": int(y_threat.sum()),
            "detected": int(((y_thr_p == 1) & (y_threat == 1)).sum()),
        }
    return out


def civil_spam_proxy(df: Any, texts: list[str]) -> np.ndarray:
    """Proxy spam gold: URL + promo cues (Civil has no spam column)."""
    from engine.spam_cues import has_clear_spam_cue

    return np.array([1 if has_clear_spam_cue(t) else 0 for t in texts], dtype=np.int8)