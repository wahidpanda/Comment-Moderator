"""Identity term detection (audit/routing; rules layer must not ban terms alone)."""

from __future__ import annotations

import re

from engine.rules import load_rules


def mentions_identity_term(text: str) -> bool:
    terms = load_rules().get("identity_terms", [])
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text, re.I):
            return True
    return False