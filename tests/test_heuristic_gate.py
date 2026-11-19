from pathlib import Path

from engine.lr_layer import LRLayer
from engine.pipeline import ModerationPipeline
from engine.schemas import CommentInput, ModerationLabel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"


def _fixture_pipeline() -> ModerationPipeline:
    lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
    return ModerationPipeline(lr=lr, use_transformer=False)


def test_sarcastic_thanks_not_praise() -> None:
    p = _fixture_pipeline()
    r = p.analyze_one(CommentInput(id="1", text="Thanks for nothing, idiot"))
    assert r.label != ModerationLabel.PRAISE
    assert r.label in (ModerationLabel.TOXIC, ModerationLabel.NEUTRAL)


def test_toxic_question_not_question() -> None:
    p = _fixture_pipeline()
    r = p.analyze_one(CommentInput(id="2", text="What kind of moron makes this?"))
    assert r.label != ModerationLabel.QUESTION


def test_identity_praise_stays_safe() -> None:
    p = _fixture_pipeline()
    r = p.analyze_one(CommentInput(id="3", text="As a gay developer I love this"))
    assert r.label in (ModerationLabel.PRAISE, ModerationLabel.NEUTRAL)
    assert r.label != ModerationLabel.IDENTITY_ATTACK