"""Comment text normalization for cache keys."""

from __future__ import annotations

import re


def normalize_comment_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t