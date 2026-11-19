from pathlib import Path

from engine.lr_layer import LRLayer
from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"


def test_indirect_threat_rules() -> None:
    lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
    p = ModerationPipeline(lr=lr, use_transformer=False)
    for text in ("Watch your back.", "I will find you", "Hope your house burns"):
        r = p.analyze_one(CommentInput(id="t", text=text))
        assert r.label == ModerationLabel.THREAT
        assert r.decided_by == "rules"
        assert r.needs_human is True