"""YAML rule layer: spam, threats, scams, duplicates. Identity terms alone never flag."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from engine.schemas import ModerationLabel, ScoreBreakdown

_RULES_PATH = Path(__file__).with_name("rules.yaml")


def load_rules() -> dict[str, Any]:
    with _RULES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


class RuleEngine:
    def __init__(self) -> None:
        self.rules = load_rules()
        self._recent_texts: deque[str] = deque(maxlen=self.rules.get("repeated", {}).get("window", 50))

    def _identity_only(self, text: str) -> bool:
        """True if text is ONLY identity term mentions (no rule should fire)."""
        terms = self.rules.get("identity_terms", [])
        lowered = text.lower().strip()
        if not lowered:
            return False
        for term in terms:
            lowered = re.sub(rf"\b{re.escape(term)}\b", " ", lowered)
        lowered = re.sub(r"[^\w\s]", " ", lowered)
        return len(lowered.split()) <= 1

    def evaluate(self, text: str, comment_id: str) -> dict[str, Any] | None:
        if self._identity_only(text):
            return None

        reasons: list[str] = []
        scores = ScoreBreakdown()
        label: ModerationLabel | None = None

        for pattern in self.rules.get("threat", {}).get("patterns", []):
            if re.search(pattern, text):
                label = ModerationLabel.THREAT
                scores.threat = 0.95
                scores.toxicity = 0.9
                reasons.append("Matched threat pattern")
                break

        for pattern in self.rules.get("spam", {}).get("patterns", []):
            if re.search(pattern, text):
                label = ModerationLabel.SPAM
                scores.spam = 0.9
                reasons.append("Matched spam pattern")
                break

        for pattern in self.rules.get("scam", {}).get("patterns", []):
            if re.search(pattern, text):
                label = ModerationLabel.SPAM
                scores.spam = 0.95
                reasons.append("Matched scam pattern")
                break

        spam_cfg = self.rules.get("spam", {}) or {}
        link_cfg = spam_cfg.get("link_heavy") or {}
        min_links = link_cfg.get("min_links", 3)
        min_len = link_cfg.get("min_length", 20)
        links = _URL_RE.findall(text)
        if len(links) >= min_links and len(text) >= min_len:
            label = label or ModerationLabel.SPAM
            scores.spam = max(scores.spam, 0.85)
            reasons.append(f"Many links ({len(links)})")

        rep = self.rules.get("repeated", {})
        norm = re.sub(r"\s+", " ", text.lower().strip())
        if norm and list(self._recent_texts).count(norm) >= rep.get("min_duplicates", 4) - 1:
            label = label or ModerationLabel.SPAM
            scores.spam = max(scores.spam, 0.8)
            reasons.append("Repeated comment in thread")
        self._recent_texts.append(norm)

        if label is None:
            return None

        risk = _risk_from_scores(scores, label)
        return {
            "label": label,
            "scores": scores,
            "risk": risk,
            "reasons": reasons,
            "needs_human": label == ModerationLabel.THREAT or label == ModerationLabel.IDENTITY_ATTACK,
        }


def _risk_from_scores(scores: ScoreBreakdown, label: ModerationLabel) -> int:
    base = max(
        scores.toxicity,
        scores.threat,
        scores.identity_attack,
        scores.spam,
        scores.insult,
    )
    if label == ModerationLabel.THREAT:
        base = max(base, 0.95)
    return int(min(100, round(base * 100)))