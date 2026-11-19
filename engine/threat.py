"""Threat scoring thresholds (LR head + Detoxify)."""

from __future__ import annotations

import os

from engine.schemas import ModerationLabel

THREAT_LABEL_THRESHOLD = float(os.getenv("THREAT_LABEL_THRESHOLD", "0.5"))
THREAT_NEEDS_HUMAN_THRESHOLD = float(os.getenv("THREAT_NEEDS_HUMAN_THRESHOLD", "0.35"))


def effective_threat_score(*scores: float | None) -> float:
    present = [float(s) for s in scores if s is not None]
    return max(present) if present else 0.0


def threat_needs_human(threat_score: float) -> bool:
    return threat_score >= THREAT_NEEDS_HUMAN_THRESHOLD


def apply_threat_label(label: ModerationLabel, threat_score: float) -> ModerationLabel:
    if threat_score >= THREAT_LABEL_THRESHOLD:
        return ModerationLabel.THREAT
    return label


def merge_needs_human(existing: bool, threat_score: float) -> bool:
    return existing or threat_needs_human(threat_score)


def threat_score_after_detoxify_scan(
    lr_threat: float | None,
    prior: float | None,
    detox_threat: float | None,
) -> tuple[float, bool]:
    """When Detoxify ran, prefer its threat head over LR false positives (e.g. idioms)."""
    if detox_threat is None:
        merged = effective_threat_score(lr_threat, prior)
        return merged, threat_needs_human(merged)
    display = max(float(detox_threat), float(prior or 0.0))
    return display, threat_needs_human(float(detox_threat))