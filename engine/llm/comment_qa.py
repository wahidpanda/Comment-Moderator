"""Ask questions about an analyzed comment set (answers cite comment ids)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from engine.llm.cache import CacheContext, cache_key, get_cached, put_cached
from engine.llm.client import LLMClient, LLMError, LLMNoEndpointsError, LLMQuotaError, LLMParseError
from engine.llm.parse import strip_llm_json_wrapper

logger = logging.getLogger(__name__)

COMMENT_QA_PROMPT_VERSION = "comment-qa-v1"
_MAX_COMMENTS = 180
_MAX_COMMENT_CHARS = 280
_MAX_QUESTION_CHARS = 500


class CommentForQA(BaseModel):
    id: str
    text: str
    creator_bucket: str | None = None


class CommentQAResult(BaseModel):
    answer: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    source: Literal["llm", "heuristic"] = "heuristic"
    llm_requests: int = 0
    cache_hit: bool = False
    budget_exhausted: bool = False
    error: str | None = None


class _QAOut(BaseModel):
    answer: str
    citation_ids: list[str] = Field(default_factory=list)


def _comments_fingerprint(comments: list[CommentForQA]) -> str:
    parts = sorted(f"{c.id}:{len(c.text)}" for c in comments[:_MAX_COMMENTS])
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _question_hash(question: str) -> str:
    norm = re.sub(r"\s+", " ", question.strip().lower())[:_MAX_QUESTION_CHARS]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _system_prompt() -> str:
    return (
        "You answer creator questions about their YouTube comments using ONLY the provided comments. "
        "Be concise (2–5 sentences). When you rely on specific comments, list their ids in citation_ids. "
        "Do not invent comments or ids. If the comments do not contain enough information, say so and "
        "cite the closest matches. Output ONLY valid JSON."
    )


def _user_payload(
    *,
    title: str,
    channel: str,
    comments: list[CommentForQA],
    question: str,
) -> str:
    lines: list[str] = []
    for c in comments[:_MAX_COMMENTS]:
        bucket = f" [{c.creator_bucket}]" if c.creator_bucket else ""
        text = c.text.replace("\n", " ").strip()[:_MAX_COMMENT_CHARS]
        lines.append(f"- id={c.id}{bucket}: {text}")
    return (
        f"Video: {title or '(unknown)'}\nChannel: {channel or '(unknown)'}\n\n"
        f"Comments ({len(lines)}):\n"
        + "\n".join(lines)
        + f"\n\nCreator question: {question.strip()[:_MAX_QUESTION_CHARS]}\n\n"
        'Return JSON: {"answer":"...","citation_ids":["id1","id2"]}'
    )


def _heuristic_answer(
    comments: list[CommentForQA],
    question: str,
) -> CommentQAResult:
    q = question.strip()
    if not q:
        return CommentQAResult(answer="Ask a question about the analyzed comments.", error="empty_question")

    tokens = [t for t in re.findall(r"[a-z0-9']+", q.lower()) if len(t) > 2]
    if not tokens:
        tokens = [t for t in q.lower().split() if t]

    scored: list[tuple[float, CommentForQA]] = []
    for c in comments:
        text_l = c.text.lower()
        hits = sum(1 for t in tokens if t in text_l)
        if hits:
            scored.append((hits / max(len(tokens), 1), c))

    scored.sort(key=lambda x: (-x[0], -len(x[1].text)))
    top = [c for _, c in scored[:8]]
    if not top:
        by_len = sorted(comments, key=lambda c: len(c.text), reverse=True)[:3]
        top = by_len

    ids = [c.id for c in top[:6]]
    snippets = "; ".join(f"[{c.id}] {c.text[:100]}" for c in top[:3])
    answer = (
        f"Heuristic search (no LLM): found {len(top)} related comment(s). "
        f"Examples: {snippets}"
    )
    return CommentQAResult(answer=answer, citation_ids=ids, source="heuristic")


def _parse_qa(content: str, valid_ids: set[str]) -> _QAOut:
    text = strip_llm_json_wrapper(content)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise LLMParseError("Expected JSON object")
    out = _QAOut.model_validate(data)
    citations = [i for i in out.citation_ids if i in valid_ids]
    if not citations and out.citation_ids:
        logger.warning("LLM cited unknown ids: %s", out.citation_ids[:5])
    return _QAOut(answer=out.answer.strip(), citation_ids=citations[:12])


def ask_about_comments(
    *,
    video_id: str,
    title: str,
    channel: str,
    comments: list[CommentForQA],
    question: str,
) -> CommentQAResult:
    q = question.strip()
    if not q:
        return CommentQAResult(error="empty_question")
    if not comments:
        return CommentQAResult(error="no_comments")

    valid_ids = {c.id for c in comments}
    fp = _comments_fingerprint(comments)
    qh = _question_hash(q)
    cache_text = f"comment-qa:{video_id}:{fp}:{qh}"

    try:
        client = LLMClient()
        models = client._resolved_models()
    except LLMError as exc:
        out = _heuristic_answer(comments, q)
        out.error = str(exc)[:200]
        return out

    if not models:
        out = _heuristic_answer(comments, q)
        out.error = "OPENROUTER model not configured"
        return out

    ctx = CacheContext(
        prompt_version=COMMENT_QA_PROMPT_VERSION,
        batch_size=1,
        model_chain="|".join(models),
    )
    key = cache_key(cache_text, ctx)
    cached = get_cached(key)
    if cached:
        try:
            parsed = _QAOut.model_validate(cached)
            citations = [i for i in parsed.citation_ids if i in valid_ids]
            return CommentQAResult(
                answer=parsed.answer,
                citation_ids=citations,
                source="llm",
                cache_hit=True,
            )
        except ValidationError:
            pass

    system = _system_prompt()
    user = _user_payload(title=title, channel=channel, comments=comments, question=q)
    llm_requests = 0
    last_error: str | None = None
    budget_exhausted = False

    for model in models:
        try:
            content, raw = client._chat(model, user, system=system)
            llm_requests += 1
            parsed = _parse_qa(content, valid_ids)
            payload: dict[str, Any] = parsed.model_dump()
            put_cached(
                key,
                cache_text,
                ctx,
                payload,
                raw_response=raw[:4000] if isinstance(raw, str) else json.dumps(raw)[:4000],
                model_used=model,
            )
            return CommentQAResult(
                answer=parsed.answer,
                citation_ids=parsed.citation_ids,
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

    out = _heuristic_answer(comments, q)
    out.llm_requests = llm_requests
    out.budget_exhausted = budget_exhausted
    out.error = last_error
    if budget_exhausted:
        out.answer = out.answer + " (OpenRouter daily budget exhausted — keyword fallback.)"
    return out