from pathlib import Path

from engine.lr_layer import LRLayer
from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput


def test_ambiguous_without_transformer_needs_human(monkeypatch) -> None:
    d = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"
    lr = LRLayer(model_path=d / "model.joblib", thresholds_path=d / "thresholds.json")
    p = ModerationPipeline(lr=lr, use_transformer=False, use_llm=False)
    monkeypatch.setattr(p.lr, "classify_band", lambda _t: ("ambiguous", 0.5))
    r = p.analyze_one(CommentInput(id="a", text="edge case comment for ambiguous band"))
    assert r.needs_human is True
    assert any("ambiguous" in x.lower() or "unavailable" in x.lower() for x in r.reasons)