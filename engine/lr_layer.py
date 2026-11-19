"""Logistic regression layer with calibrated probabilities and optional per-dimension heads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

from engine.schemas import ModerationLabel, ScoreBreakdown
from engine.threat import (
    THREAT_LABEL_THRESHOLD,
    apply_threat_label,
    merge_needs_human,
    threat_needs_human,
)

_DEFAULT_MODEL = Path("models/lr/v1/model.joblib")
_DEFAULT_THRESHOLDS = Path("models/lr/v1/thresholds.json")


def _resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path
    env = os.getenv("LR_MODEL_PATH")
    if env:
        return Path(env)
    return _DEFAULT_MODEL


class LRLayer:
    def __init__(
        self,
        model_path: Path | None = None,
        thresholds_path: Path | None = None,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        thresh_env = os.getenv("LR_THRESHOLDS_PATH")
        default_thresh = (
            self.model_path.parent / "thresholds.json"
            if (self.model_path.parent / "thresholds.json").exists()
            else _DEFAULT_THRESHOLDS
        )
        self.thresholds_path = thresholds_path or (
            Path(thresh_env) if thresh_env else default_thresh
        )
        self._artifact: Any = None
        self._harmful_model: Any = None
        self._heads: dict[str, Any] = {}
        self.t_low = 0.35
        self.t_high = 0.65
        self._load_thresholds()

    def _load_thresholds(self) -> None:
        if self.thresholds_path.exists():
            data = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
            self.t_low = float(data.get("t_low", self.t_low))
            self.t_high = float(data.get("t_high", self.t_high))

    def _ensure_model(self) -> None:
        if self._harmful_model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"LR model not found at {self.model_path}. Run: make train")
        loaded = joblib.load(self.model_path)
        self._artifact = loaded
        if isinstance(loaded, dict) and "harmful" in loaded:
            self._harmful_model = loaded["harmful"]
            self._heads = loaded.get("heads") or {}
        else:
            self._harmful_model = loaded
            self._heads = {}

    def harmful_probability(self, text: str) -> float:
        self._ensure_model()
        proba = self._harmful_model.predict_proba([text])[0, 1]
        return float(proba)

    def head_probability(self, text: str, name: str) -> float | None:
        """Calibrated P(dimension >= 0.5) when a head exists."""
        self._ensure_model()
        model = self._heads.get(name)
        if model is None:
            return None
        return float(model.predict_proba([text])[0, 1])

    def score_breakdown(self, text: str, *, harmful_p: float | None = None) -> ScoreBreakdown:
        """LR scores only — never fabricate dimension probabilities."""
        p = harmful_p if harmful_p is not None else self.harmful_probability(text)
        scores = ScoreBreakdown()
        tox_head = self.head_probability(text, "toxicity")
        scores.toxicity = tox_head if tox_head is not None else p
        for dim in ("insult", "threat", "identity_attack"):
            val = self.head_probability(text, dim)
            if val is not None:
                setattr(scores, dim, val)
        scores.spam = 0.0
        return scores

    def classify_band(self, text: str) -> tuple[str, float]:
        p = self.harmful_probability(text)
        if p <= self.t_low:
            return "safe", p
        if p >= self.t_high:
            return "harmful", p
        return "ambiguous", p

    def to_result(self, text: str, band: str, p: float) -> dict:
        scores = self.score_breakdown(text, harmful_p=p)
        label = ModerationLabel.TOXIC if band == "harmful" else ModerationLabel.NEUTRAL
        threat_p = float(scores.threat)
        if self.head_probability(text, "threat") is not None:
            threat_p = float(scores.threat)
        label = apply_threat_label(label, threat_p)
        if band == "harmful" and label != ModerationLabel.THREAT and threat_p >= THREAT_LABEL_THRESHOLD:
            label = ModerationLabel.THREAT
        risk = int(round(p * 100)) if band == "harmful" else int(round(p * 15))
        if label == ModerationLabel.THREAT:
            risk = max(risk, int(min(100, round(threat_p * 100))))
        needs = merge_needs_human(band == "harmful" and p > 0.85, threat_p)
        needs = needs or (label == ModerationLabel.THREAT and threat_needs_human(threat_p))
        reasons = [f"LR harmful probability {p:.2f} ({band})"]
        if threat_p >= THREAT_LABEL_THRESHOLD:
            reasons.append(f"LR threat head p={threat_p:.2f}")
        return {
            "label": label,
            "scores": scores,
            "risk": risk,
            "reasons": reasons,
            "needs_human": needs,
        }