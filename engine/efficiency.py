"""Analyze efficiency metrics (layer mix, LLM requests, hypothetical LLM-only cost)."""

from __future__ import annotations

import math
import os
from typing import Any

from engine.llm.cache import get_daily_requests
from engine.schemas import AnalyzeResponse


def _use_creator_buckets_llm() -> bool:
    return os.getenv("USE_CREATOR_BUCKETS_LLM", "true").lower() == "true"


def build_efficiency_block(response: AnalyzeResponse) -> dict[str, Any]:
    total = int(response.summary.get("total") or len(response.results) or 0)
    layers_raw = response.summary.get("layer_counts")
    layers: dict[str, int] = dict(layers_raw) if isinstance(layers_raw, dict) else {}

    if not layers and total > 0:
        layers = {}
        for r in response.results:
            layers[r.decided_by] = layers.get(r.decided_by, 0) + 1

    batch_size = max(1, int(os.getenv("LLM_BATCH_SIZE", "15")))
    bucket_batch = max(1, int(os.getenv("CREATOR_BUCKET_BATCH_SIZE", "80")))
    moderation_llm = int(response.summary.get("moderation_llm_requests") or 0)
    creator_llm = int(response.summary.get("creator_bucket_llm_requests") or 0)
    analyze_llm = moderation_llm + creator_llm

    hypothetical_moderation = math.ceil(total / batch_size) if total else 0
    bucket_candidates = int(response.summary.get("creator_bucket_candidates") or 0)
    if bucket_candidates <= 0 and _use_creator_buckets_llm():
        bucket_candidates = sum(1 for r in response.results if (r.creator_bucket or "") == "neutral_discussion")
    hypothetical_buckets = 0
    if bucket_candidates and _use_creator_buckets_llm():
        max_req = max(1, int(os.getenv("CREATOR_BUCKET_MAX_REQUESTS", "2")))
        hypothetical_buckets = min(max_req, math.ceil(bucket_candidates / bucket_batch))
    hypothetical_all = hypothetical_moderation + hypothetical_buckets

    layer_pct: dict[str, float] = {}
    if total > 0:
        for name, count in sorted(layers.items(), key=lambda x: -x[1]):
            layer_pct[name] = round(100.0 * count / total, 1)

    max_daily = int(os.getenv("OPENROUTER_MAX_REQUESTS_PER_DAY", "40"))
    buffer = int(os.getenv("OPENROUTER_DAILY_REQUEST_BUFFER", "5"))
    daily_used = get_daily_requests()

    n_out = max(total, 1)
    ambiguous_pct = float(response.summary.get("transformer_ambiguous_pct") or 0.0)
    threat_scan_pct = float(response.summary.get("transformer_threat_scan_pct") or 0.0)

    return {
        "total": total,
        "layer_counts": layers,
        "layer_pct": layer_pct,
        "moderation_llm_requests": moderation_llm,
        "creator_bucket_llm_requests": creator_llm,
        "llm_requests_analyze": analyze_llm,
        "llm_requests_insights": int(response.summary.get("insights_llm_requests") or 0),
        "llm_requests_ask": int(response.summary.get("ask_llm_requests") or 0),
        "llm_requests_reply": int(response.summary.get("reply_llm_requests") or 0),
        "hypothetical_llm_only_requests": hypothetical_moderation,
        "hypothetical_bucket_requests": hypothetical_buckets,
        "hypothetical_llm_only_with_buckets": hypothetical_all,
        "batch_size": batch_size,
        "bucket_batch_size": bucket_batch,
        "requests_saved_vs_llm_only": max(0, hypothetical_all - analyze_llm),
        "transformer_ambiguous_pct": ambiguous_pct,
        "transformer_threat_scan_pct": threat_scan_pct,
        "openrouter_daily_used": daily_used,
        "openrouter_daily_limit": max(1, max_daily - buffer),
    }


def attach_efficiency_summary(response: AnalyzeResponse) -> AnalyzeResponse:
    response.summary["efficiency"] = build_efficiency_block(response)
    return response