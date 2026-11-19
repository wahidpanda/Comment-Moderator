from engine.efficiency import build_efficiency_block
from engine.schemas import AnalyzeResponse, ModerationLabel, ModerationResult, ScoreBreakdown


def _row(decided_by: str) -> ModerationResult:
    return ModerationResult(
        id="1",
        text="hi",
        label=ModerationLabel.NEUTRAL,
        scores=ScoreBreakdown(),
        risk=5,
        decided_by=decided_by,  # type: ignore[arg-type]
    )


def test_efficiency_layer_pct_and_hypothetical() -> None:
    resp = AnalyzeResponse(
        results=[_row("lr"), _row("lr"), _row("rules"), _row("transformer")],
        summary={
            "total": 4,
            "layer_counts": {"lr": 2, "rules": 1, "transformer": 1},
            "moderation_llm_requests": 0,
            "creator_bucket_llm_requests": 1,
        },
    )
    block = build_efficiency_block(resp)
    assert block["layer_pct"]["lr"] == 50.0
    assert block["llm_requests_analyze"] == 1
    assert block["hypothetical_llm_only_with_buckets"] >= 1