from engine.llm.comment_qa import CommentForQA, ask_about_comments


def test_heuristic_ask_without_llm(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    comments = [
        CommentForQA(id="a1", text="Which microphone do you use in this video?"),
        CommentForQA(id="b2", text="Great tutorial, thanks!"),
    ]
    out = ask_about_comments(
        video_id="vid",
        title="Mic test",
        channel="Chan",
        comments=comments,
        question="What microphone questions do people ask?",
    )
    assert out.source == "heuristic"
    assert "a1" in out.citation_ids or len(out.citation_ids) > 0
    assert "microphone" in out.answer.lower() or "mic" in out.answer.lower()