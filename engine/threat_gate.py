"""When to run Detoxify for threat-only enrichment (not full ambiguous routing)."""

from __future__ import annotations

import os
import re

from engine.rules import load_rules

_SECOND_PERSON_VIOLENCE = re.compile(
    r"\b(you|your|u|ya|y'all|you're|youre)\b.{0,48}\b("
    r"kill|hurt|die|stab|shoot|murder|beat|destroy|attack|rape|burn"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

_INDIRECT_THREAT = re.compile(
    r"(?i)\b(sleep with one eye open|put you in the ground|watch your back)\b",
    re.IGNORECASE,
)


def threat_scan_lr_gate() -> float:
    return float(os.getenv("THREAT_SCAN_LR_GATE", "0.20"))


def threat_pattern_hint(text: str) -> bool:
    for pattern in load_rules().get("threat", {}).get("patterns", []):
        if re.search(pattern, text):
            return True
    if _INDIRECT_THREAT.search(text):
        return True
    return bool(_SECOND_PERSON_VIOLENCE.search(text))


def should_threat_detoxify_scan(
    text: str, lr_threat: float | None, *, gate: float | None = None
) -> bool:
    if threat_pattern_hint(text):
        return True
    g = threat_scan_lr_gate() if gate is None else gate
    return lr_threat is not None and lr_threat > g