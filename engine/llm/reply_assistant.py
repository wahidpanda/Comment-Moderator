"""Style-matched reply suggestions (copy/edit only — never posts)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from engine.llm.cache import CacheContext, cache_key, get_cached, put_cached
from engine.llm.client import LLMClient, LLMError, LLMNoEndpointsError, LLMQuotaError, LLMParseError
from engine.llm.config import load_openrouter_config
from engine.llm.parse import strip_llm_json_wrapper
from engine.llm.prompt import wrap_comment

logger = logging.getLogger(__name__)

REPLY_PROMPT_VERSION = "reply-v1"
_MAX_OWNER_SAMPLES = 20
_MAX_REPLY_CHARS = 500
_BLOCKED_REPLY_BUCKETS = frozenset({"nasty", "spam_scam", "harassment"})
_BLOCKED_REPLY_LABELS = frozenset({"toxic", "threat", "identity_attack", "spam"})


class ReplyItemIn(BaseModel):
    id: str
    text: str
    creator_bucket: str | None = None


class ReplySuggestion(BaseModel):
    id: str
    suggested_reply: str
    notes: list[str] = Field(default_factory=list)


class ReplySuggestionResult(BaseModel):
    suggestions: list[ReplySuggestion]
    llm_requests: int = 0
    cache_hits: int = 0
    error: str | None = None
    budget_exhausted: bool = False


class _ReplyItemOut(BaseModel):
    id: str
    suggested_reply: str
    notes: list[str] = Field(default_factory=list)


def _owner_fingerprint(samples: list[str]) -> str:
    joined = "\n".join(samples[:_MAX_OWNER_SAMPLES])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _system_prompt(channel: str) -> str:
    return (
        f"You draft YouTube comment replies for the channel “{channel}”. "
        "Match the creator’s tone using ONLY the owner reply examples provided. "
        "Replies must be short (1–3 sentences), friendly, and safe. "
        "Never insult, never promise actions you cannot verify, never post links unless examples do. "
        "Output ONLY valid JSON."
    )


def _user_payload(owner_samples: list[str], blocks: str) -> str:
    examples = "\n".join(f"- {t[:400]}" for t in owner_samples[:_MAX_OWNER_SAMPLES]) or "- (no owner examples)"
    return (
        "Owner reply examples from this video:\n"
        f"{examples}\n\n"
        "For each <comment> below, write suggested_reply the creator could paste.\n"
        'JSON: {"items":[{"id":"...","suggested_reply":"...","notes":["..."]}]}\n\n'
        f"{blocks}"
    )


def is_reply_eligible(item: ReplyItemIn, *, label: str | None = None) -> bool:
    if label and label in _BLOCKED_REPLY_LABELS:
        return False
    if item.creator_bucket in _BLOCKED_REPLY_BUCKETS:
        return False
    return True


def suggest_replies(
    *,
    channel_name: str,
    owner_samples: list[str],
    items: list[ReplyItemIn],
    batch_size: int | None = None,
) -> ReplySuggestionResult:
    if not items:
        return ReplySuggestionResult(suggestions=[])
    items = [i for i in items if is_reply_eligible(i)]
    if not items:
        return ReplySuggestionResult(suggestions=[], error="no_eligible_comments")
    if not owner_samples:
        return ReplySuggestionResult(
            suggestions=[
                ReplySuggestion(
                    id=i.id,
                    suggested_reply="",
                    notes=["No owner replies found on this video to learn style from."],
                )
                for i in items
            ],
            error="no_owner_samples",
        )

    try:
        client = LLMClient()
    except LLMError as exc:
        return ReplySuggestionResult(suggestions=[], error=str(exc))

    cfg = load_openrouter_config()
    models = client._resolved_models()

    fp = _owner_fingerprint(owner_samples)
    ctx = CacheContext(
        prompt_version=f"{REPLY_PROMPT_VERSION}:{fp}",
        batch_size=cfg.batch_size,
        model_chain="|".join(models),
    )
    chunk_bs = batch_size or min(cfg.batch_size, max(len(items), 1))
    system = _system_prompt(channel_name or "the creator")
    out: dict[str, ReplySuggestion] = {}
    llm_requests = 0
    cache_hits = 0
    budget_exhausted = False
    last_error: str | None = None

    uncached: list[ReplyItemIn] = []
    for item in items:
        row = get_cached(cache_key(item.text, ctx))
        if row and row.get("suggested_reply"):
            cache_hits += 1
            out[item.id] = ReplySuggestion(
                id=item.id,
                suggested_reply=str(row["suggested_reply"])[:_MAX_REPLY_CHARS],
                notes=list(row.get("notes") or ["cached"]),
            )
        else:
            uncached.append(item)

    if uncached and not budget_exhausted:
        chunk = uncached[:chunk_bs]
        blocks = "\n\n".join(wrap_comment(c.id, c.text) for c in chunk)
        user = _user_payload(owner_samples, blocks)
        expected = {c.id for c in chunk}
        parsed: list[_ReplyItemOut] | None = None
        model = models[0] if models else ""
        for model in models:
            try:
                content, raw = client._chat(model, user, system=system)
                llm_requests += 1
                parsed = _parse_reply_items(content, expected)
                break
            except LLMQuotaError as exc:
                budget_exhausted = True
                last_error = str(exc)[:200]
                break
            except LLMNoEndpointsError as exc:
                last_error = str(exc)[:200]
                break
            except (LLMParseError, LLMError, ValidationError, json.JSONDecodeError) as exc:
                last_error = str(exc)[:200]
                continue
        if budget_exhausted:
            for item in uncached:
                out[item.id] = ReplySuggestion(
                    id=item.id,
                    suggested_reply="",
                    notes=["Skipped: daily OpenRouter budget exhausted."],
                )
        elif parsed is None:
            for item in uncached:
                out[item.id] = ReplySuggestion(
                    id=item.id,
                    suggested_reply="",
                    notes=[last_error or "LLM unavailable for reply suggestions"],
                )
        else:
            raw_json = json.dumps({"model": model})
            for item in parsed:
                text = item.suggested_reply.strip()[:_MAX_REPLY_CHARS]
                sug = ReplySuggestion(id=item.id, suggested_reply=text, notes=item.notes or [])
                out[item.id] = sug
                comment = next(c for c in chunk if c.id == item.id)
                put_cached(
                    cache_key(comment.text, ctx),
                    comment.text,
                    ctx,
                    {"suggested_reply": text, "notes": sug.notes},
                    raw_response=raw_json,
                    model_used=model,
                )
            for item in uncached[len(chunk) :]:
                out[item.id] = ReplySuggestion(
                    id=item.id,
                    suggested_reply="",
                    notes=["Skipped: batched in one request; narrow selection to fit batch."],
                )

    ordered = [out[i.id] for i in items if i.id in out]
    return ReplySuggestionResult(
        suggestions=ordered,
        llm_requests=llm_requests,
        cache_hits=cache_hits,
        error=last_error,
        budget_exhausted=budget_exhausted,
    )


def _parse_reply_items(content: str, expected_ids: set[str]) -> list[_ReplyItemOut]:
    text = strip_llm_json_wrapper(content)
    data = json.loads(text)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise LLMParseError("Expected JSON array or {items: [...]}")
    items = [_ReplyItemOut.model_validate(x) for x in data]
    got = {x.id for x in items}
    if got != expected_ids:
        raise LLMParseError(f"ID mismatch expected={expected_ids} got={got}")
    return items