"""Optional YouTube-spam classifier (UCI Spam Collection)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib

from engine.schemas import ModerationLabel, ScoreBreakdown
from engine.spam_cues import spam_cue_reasons

_DEFAULT_MODEL = Path("models/spam/v1/model.joblib")
_DEFAULT_META = Path("models/spam/v1/metrics.json")


class SpamLayer:
    def __init__(
        self,
        model_path: Path | None = None,
        *,
        enabled: bool = True,
        threshold: float | None = None,
    ) -> None:
        env = os.getenv("SPAM_MODEL_PATH")
        self.model_path = model_path or (Path(env) if env else _DEFAULT_MODEL)
        self.enabled = enabled and self.model_path.exists()
        self._model: Any = None
        self.threshold = threshold if threshold is not None else self._load_threshold()

    def _load_threshold(self) -> float:
        meta = self.model_path.parent / "metrics.json"
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return float(data.get("spam_threshold", 0.55))
        return 0.55

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            self.enabled = False
            return
        self._model = joblib.load(self.model_path)

    def spam_probability(self, text: str) -> float | None:
        if not self.enabled:
            return None
        self._ensure()
        if self._model is None:
            return None
        return float(self._model.predict_proba([text])[0, 1])

    def evaluate(self, text: str) -> dict[str, Any] | None:
        p = self.spam_probability(text)
        if p is None or p < self.threshold:
            return None
        cues = spam_cue_reasons(text)
        if not cues and p < 0.95:
            return None
        scores = ScoreBreakdown(spam=p)
        reason_parts = [f"Spam classifier p={p:.2f}"]
        if cues:
            reason_parts.append("cues=" + ",".join(cues))
        return {
            "label": ModerationLabel.SPAM,
            "scores": scores,
            "risk": int(min(100, round(p * 100))),
            "reasons": reason_parts,
            "needs_human": False,
        }