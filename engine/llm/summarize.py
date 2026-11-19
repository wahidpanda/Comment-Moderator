"""OpenRouter plain-text video summarization (optional)."""

from __future__ import annotations

import logging
import os

import httpx

from engine.llm.budget import BudgetGuard
from engine.llm.config import load_openrouter_config

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "You summarize YouTube videos for moderators who will review comments. "
    "Write 3–5 clear sentences: topic, audience, tone, and any controversy or safety notes. "
    "Plain text only, no markdown or bullet lists."
)

_DEFAULT_SUMMARY_MODEL = "meta-llama/llama-3.2-3b-instruct:free"


def _extractive_summary(title: str, description: str, *, max_chars: int = 480) -> str:
    body = " ".join(description.split())
    if not body:
        return title
    if len(body) <= max_chars:
        return body[:max_chars]
    return body[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _pick_model() -> str | None:
    explicit = os.getenv("OPENROUTER_SUMMARY_MODEL", "").strip()
    if explicit:
        return explicit
    try:
        cfg = load_openrouter_config()
        if cfg.models:
            return cfg.primary_model
    except ValueError:
        pass
    cfg = load_openrouter_config()
    if cfg.api_key:
        return _DEFAULT_SUMMARY_MODEL
    return None


def summarize_video_llm(title: str, description: str, channel: str) -> tuple[str, str]:
    """Return (summary text, source: llm | extractive)."""
    extractive = _extractive_summary(title, description)
    model = _pick_model()
    cfg = load_openrouter_config()
    if not cfg.api_key or not model:
        return extractive, "extractive"

    desc = description[:6000] if description else ""
    user = (
        f"Channel: {channel}\nTitle: {title}\n\nDescription:\n{desc or '(no description)'}\n\n"
        "Summarize this video."
    )
    budget = BudgetGuard(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        max_requests_per_day=cfg.max_requests_per_day,
        buffer=cfg.daily_request_buffer,
        rpm=cfg.requests_per_minute,
        log_key_once=cfg.log_key_response_once,
    )
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    if cfg.app_name:
        headers["X-Title"] = cfg.app_name
    if cfg.http_referer:
        headers["HTTP-Referer"] = cfg.http_referer
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 320,
        "provider": {"data_collection": cfg.data_collection},
    }
    try:
        budget.check_can_request()
        budget.record_request()
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{cfg.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
        if resp.status_code >= 400:
            logger.warning("OpenRouter summary failed: %s", resp.text[:300])
            return extractive, "extractive"
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if len(content) < 20:
            return extractive, "extractive"
        return content, "llm"
    except Exception as exc:
        logger.warning("Video LLM summary error: %s", exc)
        return extractive, "extractive"