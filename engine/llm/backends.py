"""LLM backend selection: OpenRouter (cloud) vs local OpenAI-compatible (Ollama / LM Studio)."""

from __future__ import annotations

import logging
import os
from typing import Literal

import httpx

from engine.llm.config import (
    OpenRouterConfig,
    load_openrouter_config,
    resolve_openrouter_bucket_models,
    resolve_openrouter_models,
)

logger = logging.getLogger(__name__)

LLMBackend = Literal["openrouter", "local"]


def llm_backend() -> LLMBackend:
    raw = os.getenv("LLM_BACKEND", "openrouter").strip().lower()
    if raw in ("local", "ollama", "lmstudio", "lm_studio"):
        return "local"
    return "openrouter"


def local_llm_base_url() -> str:
    explicit = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    legacy = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
    if legacy.endswith("/v1"):
        return legacy
    return f"{legacy}/v1"


def local_llm_model() -> str:
    return (
        os.getenv("LOCAL_LLM_MODEL", "").strip()
        or os.getenv("OLLAMA_MODEL", "").strip()
        or "llama3.1:8b"
    )


def local_llm_bucket_model() -> str:
    return os.getenv("LOCAL_LLM_BUCKET_MODEL", "").strip() or local_llm_model()


def local_llm_api_key() -> str:
    return os.getenv("LOCAL_LLM_API_KEY", "ollama").strip() or "ollama"


def resolve_llm_moderation_models(config: OpenRouterConfig | None = None) -> tuple[str, ...]:
    if llm_backend() == "local":
        return (local_llm_model(),)
    return resolve_openrouter_models(config)


def resolve_llm_bucket_models(config: OpenRouterConfig | None = None) -> tuple[str, ...]:
    if llm_backend() == "local":
        return (local_llm_bucket_model(),)
    return resolve_openrouter_bucket_models(config)


def llm_configured() -> bool:
    if llm_backend() == "local":
        return probe_local_llm(timeout_s=2.0)
    cfg = load_openrouter_config()
    return bool(cfg.api_key) and bool(resolve_openrouter_models(cfg))


def probe_local_llm(*, timeout_s: float = 3.0) -> bool:
    root = local_llm_base_url().removesuffix("/v1").rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"{root}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def llm_status_line() -> str:
    if llm_backend() == "local":
        ok = probe_local_llm(timeout_s=2.0)
        return f"local {local_llm_model()} @ {local_llm_base_url()} (up={ok})"
    cfg = load_openrouter_config()
    return f"openrouter models={list(resolve_openrouter_models(cfg))} key_set={bool(cfg.api_key)}"