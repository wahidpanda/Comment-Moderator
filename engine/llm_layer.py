"""OpenRouter LLM moderation layer."""

from __future__ import annotations

import logging

from engine.llm.client import (
    LLMClient,
    LLMError,
    LLMNoEndpointsError,
    LLMQuotaError,
)
from engine.schemas import CommentInput, ModerationLabel, ScoreBreakdown

logger = logging.getLogger(__name__)


class LLMLayer:
    def __init__(self, *, enabled: bool = True, client: LLMClient | None = None) -> None:
        self.enabled = enabled
        self._client: LLMClient | None = client

    def _client_or_raise(self) -> LLMClient:
        if self._client is not None:
            return self._client
        try:
            self._client = LLMClient()
            return self._client
        except LLMError as exc:
            raise RuntimeError(str(exc)) from exc

    def predict_batch(
        self,
        comments: list[CommentInput],
        *,
        batch_size: int | None = None,
    ) -> list[dict]:
        rows, _requests = self.predict_batch_with_usage(comments, batch_size=batch_size)
        return rows

    def predict_batch_with_usage(
        self,
        comments: list[CommentInput],
        *,
        batch_size: int | None = None,
    ) -> tuple[list[dict], int]:
        if not self.enabled:
            return [self._disabled_item(c) for c in comments], 0
        try:
            client = self._client_or_raise()
            rows, metas = client.moderate_batch(comments, batch_size=batch_size)
            requests = sum(1 for m in metas if not m.from_cache)
            return [_to_pred(r, c.text) for r, c in zip(rows, comments)], requests
        except (LLMQuotaError, LLMNoEndpointsError, LLMError, RuntimeError) as exc:
            logger.warning("LLM batch failed: %s", exc)
            raise RuntimeError("llm unavailable") from exc

    def predict(self, comment: CommentInput) -> dict:
        return self.predict_batch([comment])[0]

    def _disabled_item(self, comment: CommentInput) -> dict:
        return {
            "label": ModerationLabel.NEUTRAL,
            "scores": ScoreBreakdown(),
            "risk": 50,
            "reasons": ["LLM disabled"],
            "needs_human": False,
        }


def _to_pred(row: dict, text: str) -> dict:
    label = row["label"]
    if isinstance(label, str):
        label = ModerationLabel(label)
    scores = ScoreBreakdown()
    return {
        "label": label,
        "scores": scores,
        "risk": int(row.get("risk", 50)),
        "reasons": list(row.get("reasons") or ["OpenRouter moderation"]),
        "needs_human": bool(row.get("needs_human", False)),
    }