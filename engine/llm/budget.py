"""OpenRouter key limits + request throttling."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from engine.llm.cache import get_daily_requests, increment_daily_requests

logger = logging.getLogger(__name__)

KEY_LOG_PATH = Path("docs/cache/openrouter_key_response.json")


@dataclass
class KeyLimits:
    free_remaining: int | None
    manual_cap: int | None
    effective_remaining: int | None
    raw: dict[str, Any]


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._rpm = max(requests_per_minute, 1)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            window = 60.0
            while self._times and now - self._times[0] > window:
                self._times.popleft()
            if len(self._times) >= self._rpm:
                sleep_for = window - (now - self._times[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._times.append(time.monotonic())


class LocalBudgetGuard:
    """Rate limit only — no OpenRouter daily cap (for Ollama / local servers)."""

    def __init__(self, *, rpm: int = 60) -> None:
        self.limiter = RateLimiter(rpm)

    def fetch_key_limits(self, *, force: bool = False) -> KeyLimits:
        return KeyLimits(None, None, None, {"source": "local"})

    def check_can_request(self) -> None:
        return

    def record_request(self) -> int:
        self.limiter.wait_turn()
        return increment_daily_requests()


class BudgetGuard:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_requests_per_day: int | None,
        buffer: int,
        rpm: int,
        log_key_once: bool,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_requests_per_day = max_requests_per_day
        self.buffer = buffer
        self.limiter = RateLimiter(rpm)
        self._key_logged = KEY_LOG_PATH.exists() and not log_key_once
        self._cached_limits: KeyLimits | None = None
        self._limits_fetched_at: float = 0.0

    def fetch_key_limits(self, *, force: bool = False) -> KeyLimits:
        if (
            self._cached_limits is not None
            and not force
            and time.monotonic() - self._limits_fetched_at < 300
        ):
            return self._cached_limits
        remaining_api: int | None = None
        raw: dict[str, Any] = {}
        if self.api_key:
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(
                        f"{self.base_url}/key",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    resp.raise_for_status()
                    raw = resp.json()
                if not self._key_logged:
                    KEY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    KEY_LOG_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")
                    logger.info("Wrote OpenRouter key response to %s", KEY_LOG_PATH)
                    self._key_logged = True
                data = raw.get("data") if isinstance(raw, dict) else {}
                if isinstance(data, dict):
                    fmd = data.get("free_model_daily_requests")
                    if isinstance(fmd, dict) and fmd.get("remaining") is not None:
                        remaining_api = int(fmd["remaining"])
            except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("GET /api/v1/key failed: %s", exc)
        parts = [x for x in (remaining_api, self.max_requests_per_day) if x is not None]
        effective = min(parts) if parts else self.max_requests_per_day
        limits = KeyLimits(
            free_remaining=remaining_api,
            manual_cap=self.max_requests_per_day,
            effective_remaining=effective,
            raw=raw,
        )
        self._cached_limits = limits
        self._limits_fetched_at = time.monotonic()
        return limits

    def check_can_request(self) -> None:
        from engine.llm.client import LLMQuotaError

        limits = self.fetch_key_limits()
        local_used = get_daily_requests()
        if limits.effective_remaining is None:
            return
        if local_used + self.buffer >= limits.effective_remaining:
            raise LLMQuotaError(
                f"Daily request budget exhausted (local={local_used}, cap={limits.effective_remaining})"
            )

    def record_request(self) -> int:
        self.limiter.wait_turn()
        return increment_daily_requests()