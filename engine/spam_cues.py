"""Deterministic spam cues (gating before ML spam label)."""

from __future__ import annotations

import re

from engine.rules import load_rules

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def spam_cue_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    rules = load_rules()
    for pattern in rules.get("spam", {}).get("patterns", []):
        if re.search(pattern, text):
            reasons.append("rule_spam_pattern")
            break
    for pattern in rules.get("scam", {}).get("patterns", []):
        if re.search(pattern, text):
            reasons.append("rule_scam_pattern")
            break
    links = _URL_RE.findall(text)
    link_cfg = (rules.get("spam", {}) or {}).get("link_heavy") or {}
    if len(links) >= int(link_cfg.get("min_links", 3)) and len(text) >= int(
        link_cfg.get("min_length", 20)
    ):
        reasons.append("link_heavy")
    if re.search(r"\b(buy now|click here|free money|subscribe|telegram|crypto|invest)\b", text, re.I):
        reasons.append("promo_keyword")
    return reasons


def has_clear_spam_cue(text: str) -> bool:
    return bool(spam_cue_reasons(text))