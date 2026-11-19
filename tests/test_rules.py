from engine.rules import RuleEngine
from engine.schemas import ModerationLabel


def test_identity_terms_alone_do_not_flag() -> None:
    eng = RuleEngine()
    assert eng.evaluate("gay", "1") is None
    assert eng.evaluate("muslim", "2") is None


def test_threat_pattern() -> None:
    eng = RuleEngine()
    hit = eng.evaluate("I will kill you if you post again", "3")
    assert hit is not None
    assert hit["label"] == ModerationLabel.THREAT
    assert hit["needs_human"] is True


def test_spam_pattern() -> None:
    eng = RuleEngine()
    hit = eng.evaluate("buy now click here free money", "4")
    assert hit is not None
    assert hit["label"] == ModerationLabel.SPAM