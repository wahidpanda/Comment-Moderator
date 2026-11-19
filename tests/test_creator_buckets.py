from engine.creator_buckets import filter_owner_comments, heuristic_bucket_from_result
from engine.schemas import CommentInput, ModerationLabel, ModerationResult, ScoreBreakdown


def test_filter_owner_comments() -> None:
    comments = [
        CommentInput(id="a", text="hi", author_channel_id="UCowner"),
        CommentInput(id="b", text="question?", author_channel_id="UCviewer"),
        CommentInput(id="c", text="no id"),
    ]
    kept, n = filter_owner_comments(comments, owner_channel_id="UCowner")
    assert n == 1
    assert [c.id for c in kept] == ["b", "c"]


def test_heuristic_bucket_toxic_harassment() -> None:
    r = ModerationResult(
        id="1",
        text="x",
        label=ModerationLabel.TOXIC,
        scores=ScoreBreakdown(insult=0.6),
        risk=80,
        decided_by="lr",
    )
    assert heuristic_bucket_from_result(r) == "harassment"


def test_heuristic_neutral_default_discussion() -> None:
    r = ModerationResult(
        id="1",
        text="thoughts?",
        label=ModerationLabel.NEUTRAL,
        scores=ScoreBreakdown(),
        risk=5,
        decided_by="lr",
    )
    assert heuristic_bucket_from_result(r) == "neutral_discussion"