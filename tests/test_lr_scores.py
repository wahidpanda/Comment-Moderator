from pathlib import Path

from engine.lr_layer import LRLayer

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"


def test_lr_does_not_fabricate_insult() -> None:
    lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
    scores = lr.score_breakdown("hello world")
    assert abs(scores.insult - scores.toxicity * 0.8) > 1e-6 or scores.insult == 0.0
    plain = lr.to_result("hello", "safe", 0.1)
    assert plain["scores"].insult != 0.1 * 0.8