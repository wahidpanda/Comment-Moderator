"""One-shot LLM comment insights (themes, questions, requests, sentiment)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from engine.llm.cache import CacheContext, cache_key, get_cached, put_cached
from engine.llm.client import LLMClient, LLMError, LLMNoEndpointsError, LLMQuotaError, LLMParseError
from engine.llm.parse import strip_llm_json_wrapper

logger = logging.getLogger(__name__)

INSIGHTS_PROMPT_VERSION = "insights-v1"
_MAX_COMMENTS = 180
_MAX_COMMENT_CHARS = 280


class CommentForInsights(BaseModel):
    id: str
    text: str
    creator_bucket: str | None = None


class ClusteredQuestion(BaseModel):
    theme: str
    examples: list[str] = Field(default_factory=list)
    count_estimate: int | None = None


class SentimentInsight(BaseModel):
    overall: Literal["positive", "mixed", "negative", "neutral"] = "neutral"
    summary: str = ""


class CommentInsightsResult(BaseModel):
    themes: list[str] = Field(default_factory=list)
    questions: list[ClusteredQuestion] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    sentiment: SentimentInsight = Field(default_factory=SentimentInsight)
    source: Literal["llm", "heuristic"] = "heuristic"
    llm_requests: int = 0
    cache_hit: bool = False
    budget_exhausted: bool = False
    error: str | None = None


class _InsightsOut(BaseModel):
    themes: list[str] = Field(default_factory=list)
    questions: list[ClusteredQuestion] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    sentiment: SentimentInsight = Field(default_factory=SentimentInsight)


def _comments_fingerprint(comments: list[CommentForInsights]) -> str:
    parts = sorted(f"{c.id}:{len(c.text)}" for c in comments[:_MAX_COMMENTS])
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _system_prompt() -> str:
    return (
        "You analyze YouTube comment sections for creators. "
        "Cluster similar questions together. Capture viewer requests and next-video ideas. "
        "Be concise and practical. Output ONLY valid JSON matching the schema."
    )


def _user_payload(
    *,
    title: str,
    channel: str,
    comments: list[CommentForInsights],
    bucket_counts: dict[str, int] | None,
) -> str:
    lines: list[str] = []
    for c in comments[:_MAX_COMMENTS]:
        bucket = f" [{c.creator_bucket}]" if c.creator_bucket else ""
        text = c.text.replace("\n", " ").strip()[:_MAX_COMMENT_CHARS]
        lines.append(f"- id={c.id}{bucket}: {text}")
    bucket_line = ""
    if bucket_counts:
        bucket_line = "Moderation bucket counts: " + ", ".join(
            f"{k}={v}" for k, v in sorted(bucket_counts.items(), key=lambda x: -x[1])
        )
    return (
        f"Video: {title or '(unknown)'}\nChannel: {channel or '(unknown)'}\n"
        f"{bucket_line}\n\n"
        f"Comments ({len(lines)} shown):\n"
        + "\n".join(lines)
        + "\n\n"
        'Return JSON: {"themes":["..."],"questions":[{"theme":"...","examples":["quote",...],'
        '"count_estimate":3}],"requests":["next video idea",...],'
        '"sentiment":{"overall":"positive|mixed|negative|neutral","summary":"1-2 sentences"}}'
    )


def _heuristic_insights(
    comments: list[CommentForInsights],
    bucket_counts: dict[str, int] | None,
) -> CommentInsightsResult:
    counts = dict(bucket_counts or {})
    if not counts:
        for c in comments:
            b = c.creator_bucket or "neutral"
            counts[b] = counts.get(b, 0) + 1

    top_buckets = sorted(counts.items(), key=lambda x: -x[1])[:4]
    themes = [f"Mostly {name.replace('_', ' ')} ({n})" for name, n in top_buckets if n > 0]
    if not themes:
        themes = ["General discussion"]

    question_lines = [
        c.text.strip()[:160]
        for c in comments
        if (c.creator_bucket or "") == "question" or "?" in c.text
    ][:8]
    questions = (
        [ClusteredQuestion(theme="Common questions", examples=question_lines[:5], count_estimate=len(question_lines))]
        if question_lines
        else []
    )

    critique = [c.text.strip()[:120] for c in comments if (c.creator_bucket or "") in ("critical", "question")][:5]
    requests = critique[:3] if critique else []

    pos = counts.get("appreciation", 0)
    neg = counts.get("nasty", 0) + counts.get("harassment", 0) + counts.get("spam_scam", 0)
    crit = counts.get("critical", 0)
    total = max(sum(counts.values()), len(comments), 1)
    if pos > total * 0.45 and neg < total * 0.08:
        overall: Literal["positive", "mixed", "negative", "neutral"] = "positive"
    elif neg > total * 0.15:
        overall = "negative"
    elif crit > total * 0.2 or neg > total * 0.08:
        overall = "mixed"
    else:
        overall = "neutral"
    summary = (
        f"Appreciation {pos}, critical {crit}, negative/toxic {neg} of {total} classified comments "
        "(heuristic — run with OpenRouter for deeper themes)."
    )

    return CommentInsightsResult(
        themes=themes[:6],
        questions=questions,
        requests=requests,
        sentiment=SentimentInsight(overall=overall, summary=summary),
        source="heuristic",
    )


def _parse_insights(content: str) -> _InsightsOut:
    text = strip_llm_json_wrapper(content)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise LLMParseError("Expected JSON object")
    return _InsightsOut.model_validate(data)


def generate_comment_insights(
    *,
    video_id: str,
    title: str,
    channel: str,
    comments: list[CommentForInsights],
    bucket_counts: dict[str, int] | None = None,
) -> CommentInsightsResult:
    if not comments:
        return CommentInsightsResult(error="no_comments")

    fp = _comments_fingerprint(comments)
    cache_text = f"insights:{video_id}:{fp}"
    try:
        client = LLMClient()
        models = client._resolved_models()
    except LLMError as exc:
        out = _heuristic_insights(comments, bucket_counts)
        out.error = str(exc)[:200]
        return out

    if not models:
        out = _heuristic_insights(comments, bucket_counts)
        out.error = "OPENROUTER model not configured"
        return out

    ctx = CacheContext(
        prompt_version=INSIGHTS_PROMPT_VERSION,
        batch_size=1,
        model_chain="|".join(models),
    )
    key = cache_key(cache_text, ctx)
    cached = get_cached(key)
    if cached:
        try:
            parsed = _InsightsOut.model_validate(cached)
            return CommentInsightsResult(
                themes=parsed.themes[:12],
                questions=parsed.questions[:15],
                requests=parsed.requests[:12],
                sentiment=parsed.sentiment,
                source="llm",
                cache_hit=True,
            )
        except ValidationError:
            pass

    system = _system_prompt()
    user = _user_payload(title=title, channel=channel, comments=comments, bucket_counts=bucket_counts)
    llm_requests = 0
    last_error: str | None = None
    budget_exhausted = False

    for model in models:
        try:
            content, raw = client._chat(model, user, system=system)
            llm_requests += 1
            parsed = _parse_insights(content)
            payload: dict[str, Any] = parsed.model_dump()
            put_cached(
                key,
                cache_text,
                ctx,
                payload,
                raw_response=raw[:4000],
                model_used=model,
            )
            return CommentInsightsResult(
                themes=parsed.themes[:12],
                questions=parsed.questions[:15],
                requests=parsed.requests[:12],
                sentiment=parsed.sentiment,
                source="llm",
                llm_requests=llm_requests,
            )
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

    out = _heuristic_insights(comments, bucket_counts)
    out.llm_requests = llm_requests
    out.budget_exhausted = budget_exhausted
    out.error = last_error
    if budget_exhausted:
        out.sentiment.summary = (out.sentiment.summary or "") + " OpenRouter daily budget exhausted."
    return out