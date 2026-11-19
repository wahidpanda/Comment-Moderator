"""Live Detoxify acceptance (skipped when transformer extra not installed)."""

from pathlib import Path

import pytest

from engine.pipeline import ModerationPipeline
from engine.lr_layer import LRLayer
from engine.schemas import CommentInput

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "models" / "lr" / "v1"

detoxify = pytest.importorskip("detoxify")

@pytest.fixture
def pipeline() -> ModerationPipeline:
  lr = LRLayer(model_path=FIXTURE_DIR / "model.joblib", thresholds_path=FIXTURE_DIR / "thresholds.json")
  return ModerationPipeline(lr=lr, use_transformer=True, use_llm=False)


def test_threat_scan_acceptance_strings(pipeline: ModerationPipeline) -> None:
  texts = [
      ("1", "You'd better sleep with one eye open"),
      ("2", "Someone should put you in the ground"),
      ("3", "this video killed me lol"),
      ("4", "Great video!"),
  ]
  out = pipeline.analyze_batch([CommentInput(id=i, text=t) for i, t in texts])
  by_id = {r.id: r for r in out.results}
  assert by_id["1"].needs_human is True
  assert any("Detoxify threat scan=" in x for x in by_id["1"].reasons)
  assert by_id["2"].needs_human is True
  assert by_id["2"].scores.threat >= 0.35
  assert by_id["3"].needs_human is False
  assert by_id["4"].needs_human is False
  assert out.summary["transformer_threat_scan_pct"] == 100.0