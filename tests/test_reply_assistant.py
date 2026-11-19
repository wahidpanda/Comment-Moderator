from engine.llm.reply_assistant import suggest_replies
from engine.schemas import ReplyItemIn


def test_suggest_replies_no_owner_samples() -> None:
    out = suggest_replies(
        channel_name="Test Channel",
        owner_samples=[],
        items=[ReplyItemIn(id="1", text="Great video!")],
    )
    assert out.error == "no_owner_samples"
    assert out.suggestions[0].suggested_reply == ""