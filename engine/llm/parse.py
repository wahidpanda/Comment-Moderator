"""Normalize LLM JSON content before parsing."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_THINK = "think"
_THINK_BLOCK = re.compile(
    rf"<\s*{_THINK}\b[^>]*>.*?</\s*{_THINK}\s*>",
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_][\w]*)\s*:")


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


def _extract_json_value(text: str) -> str | None:
    """Return the first balanced `{...}` or `[...]` substring."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                quote = ch
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _relax_json_text(text: str) -> str:
    out = _TRAILING_COMMA.sub(r"\1", text)
    out = _UNQUOTED_KEY.sub(r'\1"\2":', out)
    return out


def _try_python_literal(text: str) -> Any | None:
    py = _relax_json_text(text)
    py = re.sub(r"\btrue\b", "True", py, flags=re.IGNORECASE)
    py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
    py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(py)
    except (SyntaxError, ValueError):
        return None


def loads_llm_json(content: str) -> Any:
    """Parse model output that should be JSON but often is slightly invalid."""
    text = strip_llm_json_wrapper(content)
    if not text:
        raise json.JSONDecodeError("empty content", text, 0)
    candidates = [text]
    blob = _extract_json_value(text)
    if blob and blob != text:
        candidates.append(blob)
    last_err: json.JSONDecodeError | None = None
    for raw in candidates:
        for variant in (raw, _relax_json_text(raw)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as exc:
                last_err = exc
            literal = _try_python_literal(variant)
            if literal is not None:
                return literal
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("could not parse LLM JSON", text, 0)