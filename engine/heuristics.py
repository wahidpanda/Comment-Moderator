"""Lightweight praise / question / critique labels."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from engine.schemas import ModerationLabel

_RULES_PATH = Path(__file__).with_name("rules.yaml")


def _load_rules() -> dict:
    with _RULES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def match_heuristic_label(text: str) -> ModerationLabel | None:
    rules = _load_rules()
    for pattern in rules.get("praise", {}).get("patterns", []):
        if re.search(pattern, text):
            return ModerationLabel.PRAISE
    q_cfg = rules.get("question", {})
    for pattern in q_cfg.get("patterns", []):
        if re.search(pattern, text.strip()):
            return ModerationLabel.QUESTION
    if q_cfg.get("require_question_mark") and "?" in text:
        return ModerationLabel.QUESTION
    if "?" in text and len(text) < 280:
        return ModerationLabel.QUESTION
    for pattern in rules.get("critique", {}).get("patterns", []):
        if re.search(pattern, text):
            return ModerationLabel.CRITIQUE
    return None