import json

from engine.llm.client import LLMClient
from engine.schemas import CommentInput


class FakeClient(LLMClient):
    def __init__(self) -> None:
        from engine.llm.config import OpenRouterConfig

        self.config = OpenRouterConfig(
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            models=("test-model",),
            batch_size=15,
            prompt_version="moderation-v1",
            max_requests_per_day=40,
            daily_request_buffer=5,
            data_collection="deny",
            zdr=False,
            requests_per_minute=20,
            http_referer=None,
            app_name=None,
            log_key_response_once=False,
        )

    def _resolved_bucket_models(self) -> tuple[str, ...]:
        return ("test-model",)

    def _chat(self, model: str, user_content: str, *, system: str):  # type: ignore[override]
        if "c1" in user_content and "c2" in user_content:
            payload = {"items": [{"id": "c1", "creator_bucket": "discussion", "reasons": ["ok"]}]}
        elif "c2" in user_content:
            payload = {"items": [{"id": "c2", "creator_bucket": "request", "reasons": ["ok"]}]}
        else:
            payload = {"items": []}
        return json.dumps(payload), {"model": model}


def test_bucket_missing_id_retry_then_discussion_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("engine.llm.client.get_cached", lambda *a, **k: None)
    monkeypatch.setattr("engine.llm.client.put_cached", lambda *a, **k: None)
    client = FakeClient()
    comments = [
        CommentInput(id="c1", text="hello"),
        CommentInput(id="c2", text="please cover kubernetes next"),
    ]
    out, reqs, inferred = client.moderate_creator_buckets(comments, batch_size=10, max_requests=3)
    assert out["c1"] == "discussion"
    assert out["c2"] == "request"
    assert reqs >= 2
    assert not inferred