"""Borderline toxicity band (Detoxify / ambiguous LR) — LLM gate helpers."""

from __future__ import annotations

import os

from engine.schemas import ModerationLabel, ModerationResult, ScoreBreakdown

BORDERLINE_LOW = float(os.getenv("BORDERLINE_TOXIC_LOW", "0.5"))
BORDERLINE_HIGH = float(os.getenv("BORDERLINE_TOXIC_HIGH", "0.85"))

_HARMFUL = frozenset(
    {
        ModerationLabel.TOXIC,
        ModerationLabel.THREAT,
        ModerationLabel.IDENTITY_ATTACK,
        ModerationLabel.SPAM,
    }
)


def harm_signal(scores: ScoreBreakdown) -> float:
    return max(scores.toxicity, scores.insult, scores.identity_attack)


def is_borderline_band(scores: ScoreBreakdown) -> bool:
    signal = harm_signal(scores)
    return BORDERLINE_LOW <= signal < BORDERLINE_HIGH


def attach_trace(result: ModerationResult, **fields: object) -> ModerationResult:
    trace = dict(result.trace or {})
    for key, value in fields.items():
        if value is not None:
            trace[key] = value
    if not trace:
        return result
    return result.model_copy(update={"trace": trace})


def apply_borderline_without_llm(
    result: ModerationResult,
    *,
    llm_skipped_reason: str | None,
    routed_via_ambiguous: bool,
) -> ModerationResult:
    """Downgrade borderline toxic/harassment labels when ambiguous routing skipped LLM."""
    if llm_skipped_reason:
        result = attach_trace(result, llm_skipped_reason=llm_skipped_reason)
    if not llm_skipped_reason or not routed_via_ambiguous:
        return result
    if not is_borderline_band(result.scores) and result.label not in (
        ModerationLabel.TOXIC,
        ModerationLabel.IDENTITY_ATTACK,
    ):
        return result
    if result.label not in _HARMFUL and harm_signal(result.scores) < BORDERLINE_HIGH:
        if is_borderline_band(result.scores):
            reasons = list(result.reasons) + ["needs review (borderline)"]
            return attach_trace(
                result.model_copy(
                    update={
                        "label": ModerationLabel.NEUTRAL,
                        "needs_human": True,
                        "reasons": reasons,
                        "risk": min(result.risk, 55),
                    }
                ),
                borderline_review=True,
                llm_skipped_reason=llm_skipped_reason,
            )
        return result
    if is_borderline_band(result.scores) or (
        result.decided_by == "transformer" and harm_signal(result.scores) < BORDERLINE_HIGH
    ):
        reasons = list(result.reasons) + ["needs review (borderline)"]
        return attach_trace(
            result.model_copy(
                update={
                    "label": ModerationLabel.NEUTRAL,
                    "needs_human": True,
                    "reasons": reasons,
                    "risk": min(result.risk, 55),
                }
            ),
            borderline_review=True,
            llm_skipped_reason=llm_skipped_reason,
        )
    return result