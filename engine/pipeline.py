"""Hybrid rules → LR → transformer pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from engine.heuristics import match_heuristic_label
from engine.identity import mentions_identity_term
from engine.lr_layer import LRLayer
from engine.rules import RuleEngine
from engine.schemas import (
    AnalyzeResponse,
    CommentInput,
    ModerationLabel,
    ModerationResult,
    ScoreBreakdown,
)
from engine.spam_layer import SpamLayer
from engine.borderline import apply_borderline_without_llm, attach_trace
from engine.llm.config import load_openrouter_config, resolve_openrouter_models
from engine.threat import (
    THREAT_NEEDS_HUMAN_THRESHOLD,
    apply_threat_label,
    effective_threat_score,
    merge_needs_human,
    threat_needs_human,
)
from engine.threat_gate import threat_pattern_hint
from engine.llm_layer import LLMLayer
from engine.transformer_layer import TransformerLayer

LayerName = Literal["rules", "lr", "transformer", "heuristic", "spam", "llm"]


@dataclass
class _TransformerPending:
    comment: CommentInput
    text: str
    prob: float
    heuristic: ModerationLabel | None


class ModerationPipeline:
    def __init__(
        self,
        *,
        lr: LRLayer | None = None,
        rules: RuleEngine | None = None,
        transformer: TransformerLayer | None = None,
        llm: LLMLayer | None = None,
        use_transformer: bool = True,
        use_llm: bool | None = None,
        llm_only: bool = False,
    ) -> None:
        self.rules = rules or RuleEngine()
        self.lr = lr or LRLayer()
        if use_llm is None:
            use_llm = os.getenv("USE_LLM", "false").lower() == "true"
        self.use_llm = use_llm
        self.llm_only = llm_only
        self.use_transformer = use_transformer
        self.transformer = transformer or TransformerLayer(enabled=use_transformer)
        self.llm = llm or LLMLayer(enabled=use_llm)
        self.spam = SpamLayer()
        self.stats = {
            "rules": 0,
            "lr": 0,
            "transformer": 0,
            "transformer_ambiguous": 0,
            "transformer_threat_scan": 0,
            "heuristic": 0,
            "spam": 0,
            "llm": 0,
        }

    def _transformer_toxicity(self, text: str) -> float | None:
        if not self.use_transformer:
            return None
        try:
            tr = self.transformer.predict(text)
            s = tr["scores"]
            return float(max(s.toxicity, s.insult, s.threat, s.identity_attack))
        except RuntimeError:
            return None

    def _heuristic_blocked(self, text: str, lr_prob: float) -> bool:
        """Block praise/question/critique shortcuts when LR or Detoxify exceed t_low."""
        if lr_prob > self.lr.t_low:
            return True
        try:
            for head in ("toxicity", "insult", "threat", "identity_attack"):
                hp = self.lr.head_probability(text, head)
                if hp is not None and hp > self.lr.t_low:
                    return True
        except FileNotFoundError:
            pass
        tox = self._transformer_toxicity(text)
        if tox is not None and tox > self.lr.t_low:
            return True
        return False

    def _empty_result(self, comment: CommentInput, text: str) -> ModerationResult:
        return ModerationResult(
            id=comment.id,
            text=text,
            label=ModerationLabel.NEUTRAL,
            scores=ScoreBreakdown(),
            risk=0,
            decided_by="rules",
            reasons=["Empty comment"],
        )

    def _ambiguous_fallback(
        self, pending: _TransformerPending, *, llm_skipped_reason: str | None = None
    ) -> ModerationResult:
        self.stats["lr"] += 1
        lr_out = self.lr.to_result(pending.text, "ambiguous", pending.prob)
        result = ModerationResult(
            id=pending.comment.id,
            text=pending.text,
            label=ModerationLabel.NEUTRAL,
            scores=lr_out["scores"],
            risk=max(lr_out["risk"], 50),
            decided_by="lr",
            reasons=lr_out["reasons"] + ["needs review (borderline)"],
            needs_human=True,
        )
        if llm_skipped_reason:
            result = attach_trace(
                result,
                llm_skipped_reason=llm_skipped_reason,
                borderline_review=True,
                routed_via_ambiguous=True,
            )
        return apply_borderline_without_llm(
            result, llm_skipped_reason=llm_skipped_reason, routed_via_ambiguous=True
        )

    def _resolve_llm_skip_reason(self, *, batch_failed: bool = False) -> str:
        if not self.use_llm:
            return "use_llm_disabled"
        from engine.llm.backends import llm_backend, llm_configured

        if llm_backend() == "local":
            if not llm_configured():
                return "local_llm_unreachable"
        else:
            cfg = load_openrouter_config()
            if not cfg.api_key:
                return "openrouter_api_key_missing"
            if not resolve_openrouter_models(cfg):
                return "openrouter_models_unconfigured"
        if batch_failed:
            return "llm_batch_failed"
        return "llm_unavailable"

    def _apply_llm(self, pending: _TransformerPending, llm_out: dict) -> ModerationResult:
        label = llm_out["label"]
        if (
            pending.heuristic
            and label == ModerationLabel.NEUTRAL
            and not self._heuristic_blocked(pending.text, pending.prob)
        ):
            label = pending.heuristic
        return attach_trace(
            ModerationResult(
                id=pending.comment.id,
                text=pending.text,
                label=label,
                scores=llm_out["scores"],
                risk=llm_out["risk"],
                decided_by="llm",
                reasons=llm_out["reasons"],
                needs_human=llm_out["needs_human"],
            ),
            llm_skipped_reason=None,
            routed_via_ambiguous=True,
        )

    def _lr_threat_score(self, text: str) -> float | None:
        try:
            return self.lr.head_probability(text, "threat")
        except FileNotFoundError:
            return None

    def _apply_transformer(
        self,
        pending: _TransformerPending,
        tr: dict,
        *,
        llm_skipped_reason: str | None = None,
    ) -> ModerationResult:
        scores = tr["scores"]
        lr_threat = self._lr_threat_score(pending.text)
        threat_merged = effective_threat_score(lr_threat, scores.threat)
        scores = scores.model_copy(update={"threat": threat_merged})
        label = apply_threat_label(tr["label"], threat_merged)
        if (
            pending.heuristic
            and label == ModerationLabel.NEUTRAL
            and not self._heuristic_blocked(pending.text, pending.prob)
        ):
            label = pending.heuristic
        reasons = list(tr["reasons"])
        if lr_threat is not None and lr_threat > 0.2:
            reasons.append(f"LR threat head p={lr_threat:.2f}")
        needs_human = merge_needs_human(tr["needs_human"], threat_merged)
        risk = max(tr["risk"], int(min(100, round(threat_merged * 100))))
        result = ModerationResult(
            id=pending.comment.id,
            text=pending.text,
            label=label,
            scores=scores,
            risk=risk,
            decided_by="transformer",
            reasons=reasons,
            needs_human=needs_human,
        )
        if llm_skipped_reason:
            result = attach_trace(
                result,
                llm_skipped_reason=llm_skipped_reason,
                routed_via_ambiguous=True,
            )
        return apply_borderline_without_llm(
            result, llm_skipped_reason=llm_skipped_reason, routed_via_ambiguous=True
        )

    def _enrich_result_threat(self, result: ModerationResult, *, detox_threat: float | None = None) -> ModerationResult:
        lr_threat = self._lr_threat_score(result.text)
        prior = float(result.scores.threat) if result.scores.threat else None
        if detox_threat is not None:
            threat_merged = float(detox_threat)
            needs_human = merge_needs_human(
                result.needs_human, threat_merged
            ) or threat_pattern_hint(result.text)
        else:
            threat_merged = effective_threat_score(lr_threat, prior)
            needs_human = merge_needs_human(
                result.needs_human, threat_merged
            ) or threat_needs_human(threat_merged)
        scores = result.scores.model_copy(update={"threat": threat_merged})
        label = apply_threat_label(result.label, threat_merged)
        risk = max(result.risk, int(min(100, round(threat_merged * 100))))
        reasons = list(result.reasons)
        if detox_threat is not None:
            reasons.append(f"Detoxify threat scan={detox_threat:.2f}")
            if (
                threat_pattern_hint(result.text)
                and detox_threat < THREAT_NEEDS_HUMAN_THRESHOLD
            ):
                reasons.append("Indirect threat phrasing (Detoxify scan)")
        elif lr_threat is not None and lr_threat > 0.2:
            reasons.append(f"LR threat head p={lr_threat:.2f}")
        return result.model_copy(
            update={
                "label": label,
                "scores": scores,
                "risk": risk,
                "reasons": reasons,
                "needs_human": needs_human,
            }
        )

    def _enrich_threat_gated(
        self, results: list[ModerationResult], transformer_batch_size: int
    ) -> list[ModerationResult]:
        if not results:
            return results
        out = list(results)
        if not self.use_transformer:
            return [self._enrich_result_threat(r, detox_threat=None) for r in out]

        texts = [r.text for r in out]
        detox_scores: list[float | None] = [None] * len(out)
        try:
            tr_out = self.transformer.predict_batch(texts, batch_size=transformer_batch_size)
            self.stats["transformer_threat_scan"] += len(texts)
            self.stats["transformer"] += len(texts)
            for i, tr in enumerate(tr_out):
                detox_scores[i] = float(tr["scores"].threat)
        except RuntimeError:
            pass

        for i, r in enumerate(out):
            if r.decided_by == "transformer" and detox_scores[i] is None:
                detox_scores[i] = float(r.scores.threat) if r.scores.threat else None
            out[i] = self._enrich_result_threat(r, detox_threat=detox_scores[i])
        return out

    def _analyze_until_transformer(
        self, comment: CommentInput
    ) -> tuple[ModerationResult | None, _TransformerPending | None]:
        """Run rules/LR; return a final result or queue for Detoxify."""
        text = comment.text.strip()
        if not text:
            return self._empty_result(comment, text), None

        if self.llm_only and self.use_llm:
            return None, _TransformerPending(
                comment=comment, text=text, prob=0.5, heuristic=None
            )

        rule_hit = self.rules.evaluate(text, comment.id)
        heuristic = match_heuristic_label(text)
        spam_hit = self.spam.evaluate(text)
        if spam_hit:
            self.stats["spam"] += 1
            return (
                ModerationResult(
                    id=comment.id,
                    text=text,
                    label=spam_hit["label"],
                    scores=spam_hit["scores"],
                    risk=spam_hit["risk"],
                    decided_by="spam",
                    reasons=spam_hit["reasons"],
                    needs_human=spam_hit["needs_human"],
                ),
                None,
            )
        if rule_hit:
            self.stats["rules"] += 1
            return (
                ModerationResult(
                    id=comment.id,
                    text=text,
                    label=rule_hit["label"],
                    scores=rule_hit["scores"],
                    risk=rule_hit["risk"],
                    decided_by="rules",
                    reasons=rule_hit["reasons"],
                    needs_human=rule_hit["needs_human"],
                ),
                None,
            )

        if heuristic in (
            ModerationLabel.PRAISE,
            ModerationLabel.QUESTION,
            ModerationLabel.CRITIQUE,
        ):
            try:
                lr_prob = self.lr.harmful_probability(text)
                if not self._heuristic_blocked(text, lr_prob):
                    self.stats["heuristic"] += 1
                    scores = self.lr.score_breakdown(text, harmful_p=lr_prob)
                    return (
                        ModerationResult(
                            id=comment.id,
                            text=text,
                            label=heuristic,
                            scores=scores,
                            risk=max(0, int(lr_prob * 20)),
                            decided_by="heuristic",
                            reasons=[f"Heuristic {heuristic.value} (LR p={lr_prob:.2f})"],
                        ),
                        None,
                    )
            except FileNotFoundError:
                if not self._heuristic_blocked(text, 0.0):
                    self.stats["heuristic"] += 1
                    return (
                        ModerationResult(
                            id=comment.id,
                            text=text,
                            label=heuristic,
                            scores=ScoreBreakdown(),
                            risk=5,
                            decided_by="heuristic",
                            reasons=[f"Heuristic {heuristic.value}"],
                        ),
                        None,
                    )

        try:
            band, prob = self.lr.classify_band(text)
        except FileNotFoundError:
            band, prob = "ambiguous", 0.5

        if (
            band == "safe"
            and mentions_identity_term(text)
            and prob < self.lr.t_high
            and (self.use_llm or self.use_transformer)
        ):
            band = "ambiguous"

        if band == "safe":
            self.stats["lr"] += 1
            label = ModerationLabel.NEUTRAL
            decided: LayerName = "lr"
            if heuristic and not self._heuristic_blocked(text, prob):
                label = heuristic
                decided = "heuristic"
                self.stats["heuristic"] += 1
            scores = self.lr.score_breakdown(text, harmful_p=prob)
            return (
                ModerationResult(
                    id=comment.id,
                    text=text,
                    label=label,
                    scores=scores,
                    risk=max(0, int(prob * 30)),
                    decided_by=decided,
                    reasons=[f"LR safe band (p={prob:.2f})"],
                ),
                None,
            )

        borderline_margin = float(os.getenv("BORDERLINE_LR_MARGIN", "0.05"))
        if band == "harmful" and self.use_llm and prob < self.lr.t_high + borderline_margin:
            return None, _TransformerPending(
                comment=comment, text=text, prob=prob, heuristic=heuristic
            )

        if band == "harmful":
            self.stats["lr"] += 1
            lr_out = self.lr.to_result(text, band, prob)
            return (
                ModerationResult(
                    id=comment.id,
                    text=text,
                    label=lr_out["label"],
                    scores=lr_out["scores"],
                    risk=lr_out["risk"],
                    decided_by="lr",
                    reasons=lr_out["reasons"],
                    needs_human=lr_out["needs_human"],
                ),
                None,
            )

        if not self.use_llm and not self.use_transformer:
            return self._ambiguous_fallback(
                _TransformerPending(comment=comment, text=text, prob=prob, heuristic=heuristic)
            ), None

        return None, _TransformerPending(
            comment=comment, text=text, prob=prob, heuristic=heuristic
        )

    def analyze_one(self, comment: CommentInput) -> ModerationResult:
        result, pending = self._analyze_until_transformer(comment)
        if result is not None:
            enriched = self._enrich_threat_gated([result], transformer_batch_size=32)
            return enriched[0]
        assert pending is not None
        skip_reason: str | None = None
        if self.use_llm:
            try:
                llm = self.llm.predict(pending.comment)
                self.stats["llm"] += 1
                out = self._apply_llm(pending, llm)
                return self._enrich_threat_gated([out], transformer_batch_size=32)[0]
            except RuntimeError:
                skip_reason = self._resolve_llm_skip_reason(batch_failed=True)
        else:
            skip_reason = self._resolve_llm_skip_reason()
        if self.use_transformer:
            try:
                tr = self.transformer.predict(pending.text)
                self.stats["transformer"] += 1
                self.stats["transformer_ambiguous"] += 1
                out = self._apply_transformer(pending, tr, llm_skipped_reason=skip_reason)
                return self._enrich_threat_gated([out], transformer_batch_size=32)[0]
            except RuntimeError:
                pass
        return self._ambiguous_fallback(pending, llm_skipped_reason=skip_reason)

    def analyze_batch(
        self, comments: list[CommentInput], *, transformer_batch_size: int = 32
    ) -> AnalyzeResponse:
        t0 = time.perf_counter()
        self.stats = {k: 0 for k in self.stats}
        pending: list[tuple[int, _TransformerPending, float]] = []
        results: list[ModerationResult | None] = [None] * len(comments)
        latencies_ms: list[float] = []

        for i, comment in enumerate(comments):
            t_comment = time.perf_counter()
            result, tr_pending = self._analyze_until_transformer(comment)
            base_ms = (time.perf_counter() - t_comment) * 1000
            if result is not None:
                results[i] = result
                latencies_ms.append(base_ms)
            elif tr_pending is not None:
                pending.append((i, tr_pending, base_ms))

        still_pending = list(pending)
        moderation_llm_requests = 0
        llm_batch_failed = False
        if still_pending and self.use_llm:
            try:
                pending_comments = [p.comment for _, p, _ in still_pending]
                t_llm = time.perf_counter()
                llm_out, moderation_llm_requests = self.llm.predict_batch_with_usage(pending_comments)
                llm_elapsed = (time.perf_counter() - t_llm) * 1000
                per_llm = llm_elapsed / max(len(pending_comments), 1)
                self.stats["llm"] += len(pending_comments)
                for (i, p, base_ms), lo in zip(still_pending, llm_out):
                    results[i] = self._apply_llm(p, lo)
                    latencies_ms.append(base_ms + per_llm)
                still_pending = []
            except RuntimeError:
                llm_batch_failed = True

        skip_reason: str | None = None
        if still_pending:
            skip_reason = self._resolve_llm_skip_reason(batch_failed=llm_batch_failed)

        if still_pending and self.use_transformer:
            try:
                texts = [p.text for _, p, _ in still_pending]
                t_tr = time.perf_counter()
                tr_out = self.transformer.predict_batch(texts, batch_size=transformer_batch_size)
                tr_elapsed_ms = (time.perf_counter() - t_tr) * 1000
                per_ms = tr_elapsed_ms / max(len(texts), 1)
                self.stats["transformer"] += len(texts)
                self.stats["transformer_ambiguous"] += len(texts)
                for (i, p, base_ms), tr in zip(still_pending, tr_out):
                    results[i] = self._apply_transformer(p, tr, llm_skipped_reason=skip_reason)
                    latencies_ms.append(base_ms + per_ms)
                still_pending = []
            except RuntimeError:
                for i, p, base_ms in still_pending:
                    results[i] = self._ambiguous_fallback(p, llm_skipped_reason=skip_reason)
                    latencies_ms.append(base_ms)
                still_pending = []
        elif still_pending:
            for i, p, base_ms in still_pending:
                results[i] = self._ambiguous_fallback(p, llm_skipped_reason=skip_reason)
                latencies_ms.append(base_ms)

        assert all(r is not None for r in results)
        final_results = [r for r in results if r is not None]
        final_results = self._enrich_threat_gated(final_results, transformer_batch_size)
        elapsed = time.perf_counter() - t0
        label_counts: dict[str, int] = {}
        for r in final_results:
            label_counts[r.label.value] = label_counts.get(r.label.value, 0) + 1
        n_out = max(len(final_results), 1)
        summary: dict[str, int | float] = {
            "total": len(final_results),
            "needs_human": sum(1 for r in final_results if r.needs_human),
            "elapsed_ms": round(elapsed * 1000, 2),
            "layer_counts": dict(self.stats),
            "moderation_llm_requests": moderation_llm_requests,
            "transformer_ambiguous_pct": round(
                100.0 * self.stats.get("transformer_ambiguous", 0) / n_out, 2
            ),
            "transformer_threat_scan_pct": round(
                100.0 * self.stats.get("transformer_threat_scan", 0) / n_out, 2
            ),
            "latency_ms_per_comment": latencies_ms,
            **{f"label_{k}": v for k, v in label_counts.items()},
        }
        if skip_reason:
            summary["moderation_llm_skipped_reason"] = skip_reason
        return AnalyzeResponse(results=final_results, summary=summary)


def pipeline_from_env() -> ModerationPipeline:
    use_transformer = os.getenv("USE_TRANSFORMER", "true").lower() == "true"
    use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    model_path = os.getenv("LR_MODEL_PATH")
    lr = LRLayer(Path(model_path)) if model_path else LRLayer()
    return ModerationPipeline(lr=lr, use_transformer=use_transformer, use_llm=use_llm)