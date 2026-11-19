"""OpenRouter LLM client (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from engine.llm.budget import BudgetGuard
from engine.llm.cache import CacheContext, cache_key, get_cached, put_cached
from engine.llm.config import (
    OpenRouterConfig,
    load_openrouter_config,
    resolve_openrouter_bucket_models,
    resolve_openrouter_models,
)
from engine.llm.parse import strip_llm_json_wrapper
from engine.llm.prompt import (
    CREATOR_BUCKETS,
    MODERATION_V3_VERSION,
    build_creator_bucket_user_payload,
    build_user_payload,
    system_prompt_for_version,
    wrap_comment,
)
from engine.schemas import CommentInput, ModerationLabel

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class LLMError(Exception):
    pass


class LLMQuotaError(LLMError):
    pass


class LLMNoEndpointsError(LLMError):
    pass


class LLMParseError(LLMError):
    pass


class ModerationItemOut(BaseModel):
    id: str
    label: str
    needs_human: bool = False
    reasons: list[str] = Field(default_factory=list)


class CreatorBucketItemOut(BaseModel):
    id: str
    creator_bucket: str
    reasons: list[str] = Field(default_factory=list)


@dataclass
class LLMCallMeta:
    model_used: str
    raw_response: str
    from_cache: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self, config: OpenRouterConfig | None = None) -> None:
        self.config = config or load_openrouter_config()
        if not self.config.api_key:
            raise LLMError("OPENROUTER_API_KEY not set")
        self._budget = BudgetGuard(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_requests_per_day=self.config.max_requests_per_day,
            buffer=self.config.daily_request_buffer,
            rpm=self.config.requests_per_minute,
            log_key_once=self.config.log_key_response_once,
        )

    def _provider_extra(self) -> dict[str, Any]:
        provider: dict[str, Any] = {"data_collection": self.config.data_collection}
        if self.config.zdr:
            provider["zdr"] = True
        return {"provider": provider}

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.http_referer:
            h["HTTP-Referer"] = self.config.http_referer
        if self.config.app_name:
            h["X-Title"] = self.config.app_name
        return h

    def _parse_items(self, content: str, expected_ids: set[str]) -> list[ModerationItemOut]:
        text = strip_llm_json_wrapper(content)
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise LLMParseError("Expected JSON array or {items: [...]}")
        items = [ModerationItemOut.model_validate(x) for x in data]
        got = {x.id for x in items}
        if got != expected_ids:
            raise LLMParseError(f"ID mismatch expected={expected_ids} got={got}")
        return items

    def _resolved_models(self) -> tuple[str, ...]:
        return resolve_openrouter_models(self.config)

    def _chat(
        self,
        model: str,
        user_content: str,
        *,
        system: str,
    ) -> tuple[str, dict[str, Any]]:
        self._budget.check_can_request()
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            **self._provider_extra(),
        }
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            self._budget.record_request()
            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(
                        f"{self.config.base_url}/chat/completions",
                        headers=self._headers(),
                        json=body,
                    )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    time.sleep(min(max(retry_after, 1.0), 60.0) * (attempt + 1))
                    last_err = LLMError(f"429 rate limited: {resp.text[:200]}")
                    continue
                if resp.status_code in (402, 403):
                    raise LLMQuotaError(resp.text[:500])
                if resp.status_code >= 400:
                    msg = resp.text[:500]
                    if "endpoint" in msg.lower() and self.config.data_collection == "deny":
                        raise LLMNoEndpointsError(msg)
                    raise LLMError(f"HTTP {resp.status_code}: {msg}")
                raw = resp.json()
                choice = raw["choices"][0]["message"]["content"]
                return choice, raw
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_err = exc
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(str(last_err) if last_err else "LLM request failed")

    def moderate_batch(
        self,
        comments: list[CommentInput],
        *,
        batch_size: int | None = None,
        prompt_version: str | None = None,
        sweep_tag: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[LLMCallMeta]]:
        if not comments:
            return [], []
        chunk_bs = batch_size if batch_size is not None else self.config.batch_size
        pv = prompt_version or self.config.prompt_version
        if sweep_tag is None and chunk_bs != self.config.batch_size:
            sweep_tag = f"sweep-bs-{chunk_bs}"
        ctx = CacheContext(
            prompt_version=pv,
            batch_size=self.config.batch_size,
            model_chain=self.config.model_chain_fingerprint,
            sweep_tag=sweep_tag,
        )
        out_map: dict[str, dict[str, Any]] = {}
        metas: list[LLMCallMeta] = []
        uncached: list[CommentInput] = []
        for c in comments:
            key = cache_key(c.text, ctx)
            hit = get_cached(key)
            if hit:
                row = dict(hit)
                lbl = row.get("label")
                if isinstance(lbl, str):
                    row["label"] = _coerce_label(lbl)
                out_map[c.id] = row
                metas.append(
                    LLMCallMeta(model_used="cache", raw_response="", from_cache=True)
                )
            else:
                uncached.append(c)

        for i in range(0, len(uncached), chunk_bs):
            chunk = uncached[i : i + chunk_bs]
            blocks = "\n\n".join(wrap_comment(c.id, c.text) for c in chunk)
            user = build_user_payload(blocks)
            expected = {c.id for c in chunk}
            parsed: list[ModerationItemOut] | None = None
            raw_json: dict[str, Any] = {}
            content = ""
            model_used = ""
            last_error: Exception | None = None
            system = system_prompt_for_version(pv)
            for model in self._resolved_models():
                try:
                    content, raw_json = self._chat(model, user, system=system)
                    parsed = self._parse_items(content, expected)
                    model_used = raw_json.get("model", model)
                    break
                except LLMNoEndpointsError:
                    raise
                except (LLMParseError, LLMError, ValidationError, json.JSONDecodeError) as exc:
                    last_error = exc
                    continue
            if parsed is None:
                raise LLMParseError(str(last_error) if last_error else "parse failed")

            usage = raw_json.get("usage") or {}
            meta = LLMCallMeta(
                model_used=model_used,
                raw_response=json.dumps(raw_json),
                from_cache=False,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            )
            metas.append(meta)
            for item in parsed:
                label = _coerce_label(item.label)
                row = {
                    "id": item.id,
                    "label": label,
                    "needs_human": item.needs_human,
                    "reasons": item.reasons or [f"OpenRouter {model_used}"],
                    "scores": {},
                    "risk": _risk_for_label(label),
                }
                comment = next(c for c in chunk if c.id == item.id)
                out_map[item.id] = row
                put_cached(
                    cache_key(comment.text, ctx),
                    comment.text,
                    ctx,
                    row,
                    raw_response=meta.raw_response,
                    model_used=model_used,
                )

        ordered = [out_map[c.id] for c in comments]
        return ordered, metas

    def _parse_creator_items(self, content: str, expected_ids: set[str]) -> list[CreatorBucketItemOut]:
        text = strip_llm_json_wrapper(content)
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise LLMParseError("Expected JSON array or {items: [...]}")
        items = [CreatorBucketItemOut.model_validate(x) for x in data]
        got = {x.id for x in items}
        if got != expected_ids:
            raise LLMParseError(f"ID mismatch expected={expected_ids} got={got}")
        return items

    def _parse_creator_items_partial(self, content: str) -> list[CreatorBucketItemOut]:
        text = strip_llm_json_wrapper(content)
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise LLMParseError("Expected JSON array or {items: [...]}")
        return [CreatorBucketItemOut.model_validate(x) for x in data]

    def _resolved_bucket_models(self) -> tuple[str, ...]:
        return resolve_openrouter_bucket_models(self.config)

    def _chat_bucket_batch(
        self,
        chunk: list[CommentInput],
        *,
        models: tuple[str, ...],
        system: str,
        ctx: CacheContext,
        out: dict[str, str],
    ) -> tuple[int, set[str]]:
        requests = 0
        inferred: set[str] = set()
        blocks = "\n\n".join(wrap_comment(c.id, c.text) for c in chunk)
        user = build_creator_bucket_user_payload(blocks)
        expected = {c.id for c in chunk}
        from engine.llm.prompt import CREATOR_NEUTRAL_BUCKETS

        valid = set(CREATOR_NEUTRAL_BUCKETS)
        parsed: list[CreatorBucketItemOut] | None = None
        last_error: Exception | None = None
        model_used = ""
        for model in models:
            try:
                content, raw_json = self._chat(model, user, system=system)
                requests += 1
                parsed = self._parse_creator_items_partial(content)
                model_used = raw_json.get("model", model)
                break
            except LLMNoEndpointsError:
                raise
            except (LLMParseError, LLMError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
        if parsed is None:
            raise LLMParseError(str(last_error) if last_error else "creator bucket parse failed")

        got: set[str] = set()
        for item in parsed:
            bucket = item.creator_bucket.strip().lower()
            if bucket not in valid:
                bucket = "discussion"
            out[item.id] = bucket
            got.add(item.id)
            comment = next((c for c in chunk if c.id == item.id), None)
            if comment:
                put_cached(
                    cache_key(comment.text, ctx),
                    comment.text,
                    ctx,
                    {"creator_bucket": bucket, "reasons": item.reasons or []},
                    raw_response=json.dumps({"model": model_used}),
                    model_used=model_used,
                )

        missing = expected - got
        if missing:
            retry = [c for c in chunk if c.id in missing]
            blocks_r = "\n\n".join(wrap_comment(c.id, c.text) for c in retry)
            user_r = build_creator_bucket_user_payload(blocks_r)
            reparsed: list[CreatorBucketItemOut] | None = None
            for model in models:
                try:
                    content, raw_json = self._chat(model, user_r, system=system)
                    requests += 1
                    reparsed = self._parse_creator_items_partial(content)
                    model_used = raw_json.get("model", model)
                    break
                except LLMNoEndpointsError:
                    raise
                except (LLMParseError, LLMError, ValidationError, json.JSONDecodeError) as exc:
                    last_error = exc
                    continue
            if reparsed:
                for item in reparsed:
                    if item.id not in missing:
                        continue
                    bucket = item.creator_bucket.strip().lower()
                    if bucket not in valid:
                        bucket = "discussion"
                    out[item.id] = bucket
                    got.add(item.id)
                    comment = next(c for c in retry if c.id == item.id)
                    put_cached(
                        cache_key(comment.text, ctx),
                        comment.text,
                        ctx,
                        {"creator_bucket": bucket, "reasons": item.reasons or []},
                        raw_response=json.dumps({"model": model_used}),
                        model_used=model_used,
                    )

        for c in chunk:
            if c.id in expected and c.id not in got:
                out[c.id] = "discussion"
                inferred.add(c.id)
        return requests, inferred

    def moderate_creator_buckets(
        self,
        comments: list[CommentInput],
        *,
        batch_size: int | None = None,
        max_requests: int | None = None,
    ) -> tuple[dict[str, str], int, set[str]]:
        if not comments:
            return {}, 0, set()
        models = self._resolved_bucket_models()
        if not self.config.api_key or not models:
            raise LLMError("OPENROUTER not configured for creator buckets")
        from engine.llm.prompt import CREATOR_NEUTRAL_BUCKETS

        valid = set(CREATOR_NEUTRAL_BUCKETS)
        out: dict[str, str] = {}
        requests = 0
        chunk_bs = batch_size if batch_size is not None else self.config.batch_size
        max_req = max_requests if max_requests is not None else 2
        system = system_prompt_for_version(MODERATION_V3_VERSION)
        ctx = CacheContext(
            prompt_version=MODERATION_V3_VERSION,
            batch_size=chunk_bs,
            model_chain=self.config.model_chain_fingerprint,
        )
        uncached: list[CommentInput] = []
        for c in comments:
            hit = get_cached(cache_key(c.text, ctx))
            if hit and hit.get("creator_bucket"):
                out[c.id] = str(hit["creator_bucket"])
            else:
                uncached.append(c)

        inferred: set[str] = set()
        for i in range(0, len(uncached), chunk_bs):
            if requests >= max_req and i > 0:
                break
            chunk = uncached[i : i + chunk_bs]
            batch_requests, batch_inferred = self._chat_bucket_batch(
                chunk, models=models, system=system, ctx=ctx, out=out
            )
            requests += batch_requests
            inferred |= batch_inferred
        return out, requests, inferred


def _coerce_label(value: str) -> ModerationLabel:
    try:
        return ModerationLabel(value.strip().lower())
    except ValueError:
        return ModerationLabel.NEUTRAL


def _risk_for_label(label: ModerationLabel) -> int:
    if label in (ModerationLabel.THREAT, ModerationLabel.IDENTITY_ATTACK):
        return 90
    if label in (ModerationLabel.TOXIC, ModerationLabel.SPAM):
        return 70
    if label == ModerationLabel.NEUTRAL:
        return 10
    return 25