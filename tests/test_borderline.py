from engine.borderline import apply_borderline_without_llm
from engine.schemas import ModerationLabel, ModerationResult, ScoreBreakdown


def test_borderline_transformer_downgrades_without_llm() -> None:
    result = ModerationResult(
        id="1",
        text="ridiculous RAG",
        label=ModerationLabel.TOXIC,
        scores=ScoreBreakdown(toxicity=0.66, insult=0.53),
        risk=66,
        decided_by="transformer",
        reasons=["Detoxify unbiased model"],
    )
    out = apply_borderline_without_llm(
        result, llm_skipped_reason="use_llm_disabled", routed_via_ambiguous=True
    )
    assert out.label == ModerationLabel.NEUTRAL
    assert out.needs_human is True
    assert out.trace.get("llm_skipped_reason") == "use_llm_disabled"