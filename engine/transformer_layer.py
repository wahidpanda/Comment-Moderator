"""Detoxify (optional) for ambiguous comments — CPU, lazy import."""

from __future__ import annotations

from engine.schemas import ModerationLabel, ScoreBreakdown
from engine.threat import apply_threat_label, merge_needs_human

_TRANSFORMER_AVAILABLE: bool | None = None
_MODEL = None


def _ensure_detoxify():
    global _TRANSFORMER_AVAILABLE, _MODEL
    if _TRANSFORMER_AVAILABLE is False:
        raise RuntimeError("detoxify not installed; pip install -e '.[transformer]'")
    if _MODEL is None:
        try:
            from detoxify import Detoxify

            _MODEL = Detoxify("unbiased")
            _TRANSFORMER_AVAILABLE = True
        except ImportError:
            _TRANSFORMER_AVAILABLE = False
            raise RuntimeError("detoxify not installed; pip install -e '.[transformer]'")
    return _MODEL


class TransformerLayer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def predict_batch(self, texts: list[str], *, batch_size: int = 16) -> list[dict]:
        if not self.enabled:
            return [self.predict(t) for t in texts]
        model = _ensure_detoxify()
        outputs: list[dict] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            raw = model.predict(chunk)
            for j in range(len(chunk)):
                row = {
                    k: float(raw[k][j]) if isinstance(raw[k], list) else float(raw[k])
                    for k in raw
                }
                outputs.append(self._from_raw(row))
        return outputs

    def _from_raw(self, raw: dict) -> dict:
        toxicity = float(raw.get("toxicity", 0))
        insult = float(raw.get("insult", 0))
        threat = float(raw.get("threat", 0))
        identity = float(raw.get("identity_attack", 0))
        scores = ScoreBreakdown(
            toxicity=toxicity,
            insult=insult,
            threat=threat,
            identity_attack=identity,
        )
        label = ModerationLabel.NEUTRAL
        reasons = ["Detoxify unbiased model"]
        label = apply_threat_label(label, threat)
        if label == ModerationLabel.THREAT:
            reasons.append(f"Detoxify threat={threat:.2f}")
        elif identity >= 0.5:
            label = ModerationLabel.IDENTITY_ATTACK
            reasons.append(f"identity_attack={identity:.2f}")
        elif toxicity >= 0.5 or insult >= 0.5:
            label = ModerationLabel.TOXIC
            reasons.append(f"toxicity={toxicity:.2f}")
        risk = int(min(100, round(max(toxicity, threat, identity, insult) * 100)))
        needs_human = merge_needs_human(
            label == ModerationLabel.IDENTITY_ATTACK,
            threat,
        )
        return {
            "label": label,
            "scores": scores,
            "risk": risk,
            "reasons": reasons,
            "needs_human": needs_human,
        }

    def predict(self, text: str) -> dict:
        if not self.enabled:
            return {
                "label": ModerationLabel.NEUTRAL,
                "scores": ScoreBreakdown(),
                "risk": 50,
                "reasons": ["Transformer disabled"],
                "needs_human": False,
            }
        model = _ensure_detoxify()
        raw = model.predict(text)
        row = {k: float(raw[k][0]) if isinstance(raw[k], list) else float(raw[k]) for k in raw}
        return self._from_raw(row)