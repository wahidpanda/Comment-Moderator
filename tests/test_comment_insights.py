from engine.llm.comment_insights import CommentForInsights, generate_comment_insights


def test_heuristic_insights_without_llm(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    comments = [
        CommentForInsights(id="1", text="Love this!", creator_bucket="appreciation"),
        CommentForInsights(id="2", text="What mic do you use?", creator_bucket="question"),
        CommentForInsights(id="3", text="Please cover Python next", creator_bucket="question"),
    ]
    out = generate_comment_insights(
        video_id="abc123",
        title="Test",
        channel="Chan",
        comments=comments,
        bucket_counts={"appreciation": 1, "question": 2},
    )
    assert out.source == "heuristic"
    assert len(out.themes) >= 1
    assert out.sentiment.overall in ("positive", "mixed", "negative", "neutral")