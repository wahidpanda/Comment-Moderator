from pathlib import Path

from engine.lr_layer import LRLayer
from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"


def _fixture_pipeline() -> ModerationPipeline:
    lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
    return ModerationPipeline(lr=lr, use_transformer=False)


def test_pipeline_threat_via_rules() -> None:
    p = _fixture_pipeline()
    r = p.analyze_one(CommentInput(id="t1", text="gonna kill you"))
    assert r.decided_by == "rules"
    assert r.label == ModerationLabel.THREAT


def test_pipeline_praise_heuristic() -> None:
    p = _fixture_pipeline()
    r = p.analyze_one(CommentInput(id="p1", text="thank you great video"))
    assert r.label == ModerationLabel.PRAISE


def test_batch_summary() -> None:
    p = _fixture_pipeline()
    resp = p.analyze_batch(
        [
            CommentInput(id="a", text="hello"),
            CommentInput(id="b", text="buy now click here free money"),
        ]
    )
    assert resp.summary["total"] == 2