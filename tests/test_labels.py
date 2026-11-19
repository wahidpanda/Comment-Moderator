import pandas as pd

from research.labels import build_unified_frame, civil_harmful_binary


def test_civil_harmful_binary() -> None:
    df = pd.DataFrame({"toxicity": [0.1, 0.9], "identity_attack": [0.0, 0.0], "threat": [0.0, 0.0]})
    y = civil_harmful_binary(df)
    assert y.tolist() == [0, 1]


def test_build_unified_frame() -> None:
    civil = pd.DataFrame({"text": ["a", "b"], "toxicity": [0.0, 0.9], "identity_attack": [0, 0], "threat": [0, 0]})
    frame = build_unified_frame(civil, None)
    assert len(frame) == 2
    assert set(frame.columns) >= {"text", "harmful", "source"}