"""Creator-facing comment buckets (moderation-v3)."""

from __future__ import annotations

import logging
import math
import os

from engine.efficiency import attach_efficiency_summary
from engine.llm.client import LLMClient, LLMError
from engine.llm.prompt import CREATOR_BUCKETS, MODERATION_V3_VERSION
from engine.schemas import AnalyzeResponse, CommentInput, ModerationLabel, ModerationResult

logger = logging.getLogger(__name__)

_VALID = set(CREATOR_BUCKETS)
_HARM_BUCKETS = {"nasty", "spam_scam", "harassment"}


def bucket_heuristic_from_result(result: ModerationResult) -> str | None:
    """Fixed bucket from moderation label / layer; None => neutral discussion needs LLM refine."""
    label = result.label
    if result.decided_by == "heuristic":
        if label == ModerationLabel.PRAISE:
            return "appreciation"
        if label == ModerationLabel.QUESTION:
            return "question"
        if label == ModerationLabel.CRITIQUE:
            return "critical"
    if label == ModerationLabel.PRAISE:
        return "appreciation"
    if label == ModerationLabel.QUESTION:
        return "question"
    if label == ModerationLabel.CRITIQUE:
        return "critical"
    if label == ModerationLabel.SPAM:
        return "spam_scam"
    if label in (ModerationLabel.THREAT, ModerationLabel.IDENTITY_ATTACK):
        return "nasty"
    if label == ModerationLabel.TOXIC:
        if result.scores.insult >= 0.45 or result.scores.identity_attack >= 0.35:
            return "harassment"
        return "nasty"
    if result.risk >= 55 and label not in (ModerationLabel.NEUTRAL,):
        return "nasty"
    if label == ModerationLabel.NEUTRAL:
        return None
    return "neutral_discussion"


def heuristic_bucket_from_result(result: ModerationResult) -> str:
    bucket = bucket_heuristic_from_result(result)
    return bucket or "neutral_discussion"


def _use_creator_llm() -> bool:
    return os.getenv("USE_CREATOR_BUCKETS_LLM", "true").lower() == "true"


def _bucket_batch_size() -> int:
    return max(15, int(os.getenv("CREATOR_BUCKET_BATCH_SIZE", "80")))


def _bucket_max_requests() -> int:
    return max(1, int(os.getenv("CREATOR_BUCKET_MAX_REQUESTS", "2")))


def apply_creator_buckets(
    response: AnalyzeResponse,
    comments: list[CommentInput],
) -> AnalyzeResponse:
    by_id = {c.id: c for c in comments}
    llm_candidates: list[CommentInput] = []
    for r in response.results:
        if bucket_heuristic_from_result(r) is not None:
            continue
        if r.label != ModerationLabel.NEUTRAL:
            continue
        if r.id in by_id:
            llm_candidates.append(by_id[r.id])

    llm_map: dict[str, str] = {}
    llm_requests = 0
    llm_attempted = False
    llm_error: str | None = None
    inferred_discussion: set[str] = set()

    if _use_creator_llm() and llm_candidates:
        llm_attempted = True
        try:
            client = LLMClient()
            llm_map, llm_requests, inferred_discussion = client.moderate_creator_buckets(
                llm_candidates,
                batch_size=_bucket_batch_size(),
                max_requests=_bucket_max_requests(),
            )
            response.summary["creator_bucket_inferred_discussion"] = len(inferred_discussion)
        except LLMError as exc:
            logger.warning("Creator bucket LLM skipped: %s", exc)
            llm_error = str(exc)[:200]
            response.summary["creator_bucket_llm_error"] = llm_error
        except Exception as exc:
            logger.warning("Creator bucket LLM failed: %s", exc)
            llm_error = str(exc)[:200]
            response.summary["creator_bucket_llm_error"] = llm_error

    bucket_counts: dict[str, int] = {b: 0 for b in CREATOR_BUCKETS}
    updated: list[ModerationResult] = []
    for r in response.results:
        trace = dict(r.trace or {})
        fixed = bucket_heuristic_from_result(r)
        if fixed:
            bucket = fixed
        elif r.id in llm_map:
            bucket = llm_map[r.id]
            if bucket not in _VALID or bucket in _HARM_BUCKETS:
                bucket = "discussion"
                trace["bucket_inferred_discussion"] = True
        elif llm_attempted:
            bucket = "neutral_discussion"
        else:
            bucket = "neutral_discussion"
        if bucket not in _VALID:
            bucket = "neutral_discussion"
        if r.id in inferred_discussion:
            bucket = "discussion"
            trace["bucket_inferred_discussion"] = True
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        updated.append(r.model_copy(update={"creator_bucket": bucket, "trace": trace}))

    response.results = updated
    response.summary["creator_bucket_counts"] = bucket_counts
    response.summary["creator_bucket_llm_requests"] = llm_requests
    response.summary["creator_bucket_candidates"] = len(llm_candidates)
    response.summary["creator_bucket_prompt"] = MODERATION_V3_VERSION
    chunk = _bucket_batch_size()
    response.summary["creator_bucket_hypothetical_requests"] = (
        min(_bucket_max_requests(), math.ceil(len(llm_candidates) / chunk)) if llm_candidates else 0
    )
    return attach_efficiency_summary(response)


def filter_owner_comments(
    comments: list[CommentInput],
    *,
    owner_channel_id: str | None,
) -> tuple[list[CommentInput], int]:
    if not owner_channel_id:
        return comments, 0
    kept: list[CommentInput] = []
    excluded = 0
    for c in comments:
        if c.author_channel_id and c.author_channel_id == owner_channel_id:
            excluded += 1
            continue
        kept.append(c)
    return kept, excluded