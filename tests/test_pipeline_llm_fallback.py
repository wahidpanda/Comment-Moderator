from pathlib import Path
from unittest.mock import MagicMock

from engine.lr_layer import LRLayer
from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel, ScoreBreakdown

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"


def test_hybrid_llm_fails_falls_back_to_transformer(monkeypatch) -> None:
    lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
    monkeypatch.setattr(lr, "classify_band", lambda _t: ("ambiguous", 0.5))

    mock_llm = MagicMock()
    mock_llm.enabled = True
    mock_llm.predict_batch.side_effect = RuntimeError("llm unavailable")
    mock_llm.predict.side_effect = RuntimeError("llm unavailable")

    mock_tr = MagicMock()
    mock_tr.enabled = True
    tr_row = {
        "label": ModerationLabel.TOXIC,
        "scores": ScoreBreakdown(toxicity=0.8, insult=0.1),
        "risk": 80,
        "reasons": ["Detoxify"],
        "needs_human": False,
    }
    mock_tr.predict.return_value = tr_row
    mock_tr.predict_batch.return_value = [tr_row]

    p = ModerationPipeline(
        lr=lr,
        llm=mock_llm,
        transformer=mock_tr,
        use_llm=True,
        use_transformer=True,
    )
    r = p.analyze_one(CommentInput(id="x", text="some edgy vague phrase here"))
    assert r.decided_by == "transformer"
    assert r.label == ModerationLabel.NEUTRAL
    assert r.needs_human is True
    assert r.trace.get("llm_skipped_reason")


def test_llm_and_transformer_fail_needs_human(monkeypatch) -> None:
    lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
    monkeypatch.setattr(lr, "classify_band", lambda _t: ("ambiguous", 0.5))

    mock_llm = MagicMock()
    mock_llm.enabled = True
    mock_llm.predict.side_effect = RuntimeError("llm unavailable")

    mock_tr = MagicMock()
    mock_tr.enabled = True
    mock_tr.predict.side_effect = RuntimeError("detoxify down")

    p = ModerationPipeline(lr=lr, llm=mock_llm, transformer=mock_tr, use_llm=True, use_transformer=True)
    r = p.analyze_one(CommentInput(id="y", text="another ambiguous comment for testing"))
    assert r.needs_human is True