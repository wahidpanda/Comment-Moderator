"""OpenRouter configuration from environment (no hardcoded models)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file() -> None:
    """Load .env into os.environ without overwriting existing keys."""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_env_file()


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str
    models: tuple[str, ...]
    batch_size: int
    prompt_version: str
    max_requests_per_day: int | None
    daily_request_buffer: int
    data_collection: str
    zdr: bool
    requests_per_minute: int
    http_referer: str | None
    app_name: str | None
    log_key_response_once: bool

    @property
    def primary_model(self) -> str:
        if not self.models:
            raise ValueError("OPENROUTER_MODEL or OPENROUTER_MODELS required")
        return self.models[0]

    @property
    def model_chain_fingerprint(self) -> str:
        return "|".join(self.models)


DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct"


def resolve_openrouter_models(config: OpenRouterConfig | None = None) -> tuple[str, ...]:
    """Configured models, else OPENROUTER_SUMMARY_MODEL, else DEFAULT_OPENROUTER_MODEL."""
    cfg = config or load_openrouter_config()
    if cfg.models:
        return cfg.models
    summary = os.getenv("OPENROUTER_SUMMARY_MODEL", "").strip()
    if summary:
        return (summary,)
    override = os.getenv("OPENROUTER_DEFAULT_MODEL", "").strip()
    if override:
        return (override,)
    return (DEFAULT_OPENROUTER_MODEL,)


def resolve_openrouter_bucket_models(config: OpenRouterConfig | None = None) -> tuple[str, ...]:
    """Models for creator-bucket pass (OPENROUTER_BUCKET_MODELS), else moderation chain."""
    cfg = config or load_openrouter_config()
    bucket = _parse_models(os.getenv("OPENROUTER_BUCKET_MODELS"))
    if bucket:
        return bucket
    return resolve_openrouter_models(cfg)


def _parse_models(raw: str | None) -> tuple[str, ...]:
    if not raw or not raw.strip():
        return ()
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return tuple(str(x).strip() for x in parsed if str(x).strip())
        except json.JSONDecodeError:
            pass
    return tuple(m.strip() for m in stripped.split(",") if m.strip())


def load_openrouter_config() -> OpenRouterConfig:
    single = os.getenv("OPENROUTER_MODEL", "").strip()
    models = _parse_models(os.getenv("OPENROUTER_MODELS"))
    if single and single not in models:
        models = (single, *models)
    max_day = os.getenv("OPENROUTER_MAX_REQUESTS_PER_DAY")
    return OpenRouterConfig(
        api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        models=models,
        batch_size=int(os.getenv("LLM_BATCH_SIZE", "15")),
        prompt_version=os.getenv("LLM_PROMPT_VERSION", "moderation-v1"),
        max_requests_per_day=int(max_day) if max_day else None,
        daily_request_buffer=int(os.getenv("OPENROUTER_DAILY_REQUEST_BUFFER", "5")),
        data_collection=os.getenv("OPENROUTER_DATA_COLLECTION", "deny"),
        zdr=os.getenv("OPENROUTER_ZDR", "false").lower() == "true",
        requests_per_minute=int(os.getenv("OPENROUTER_REQUESTS_PER_MINUTE", "20")),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER") or None,
        app_name=os.getenv("OPENROUTER_APP_NAME") or None,
        log_key_response_once=os.getenv("OPENROUTER_LOG_KEY_ONCE", "true").lower() == "true",
    )