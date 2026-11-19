"""Normalize LLM JSON content before parsing."""

from __future__ import annotations

import re

_THINK = "think"
_THINK_BLOCK = re.compile(
    rf"<\s*{_THINK}\b[^>]*>.*?</\s*{_THINK}\s*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_llm_json_wrapper(content: str) -> str:
    """Remove thinking blocks, markdown fences, and surrounding whitespace."""
    text = content.strip()
    text = _THINK_BLOCK.sub("", text)
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()