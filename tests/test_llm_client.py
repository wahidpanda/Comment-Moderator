from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from engine.llm.client import LLMClient, LLMParseError
from engine.llm.config import OpenRouterConfig
from engine.schemas import CommentInput


@pytest.fixture
def cfg() -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        models=("test/model:free", "test/fallback:free"),
        batch_size=15,
        prompt_version="moderation-v1",
        max_requests_per_day=100,
        daily_request_buffer=2,
        data_collection="deny",
        zdr=False,
        requests_per_minute=60,
        http_referer=None,
        app_name="pytest",
        log_key_response_once=False,
    )


def test_parse_and_id_validation(cfg: OpenRouterConfig) -> None:
    client = LLMClient(cfg)
    content = json.dumps(
        {
            "items": [
                {"id": "a", "label": "neutral", "needs_human": False, "reasons": ["ok"]},
                {"id": "b", "label": "toxic", "needs_human": True, "reasons": ["insult"]},
            ]
        }
    )
    items = client._parse_items(content, {"a", "b"})
    assert len(items) == 2
    with pytest.raises(LLMParseError):
        client._parse_items(content, {"a"})


def test_moderate_batch_uses_cache(cfg: OpenRouterConfig, tmp_path) -> None:
    from engine.llm import cache as cache_mod

    cache_mod.DEFAULT_DB = tmp_path / "t.sqlite"
    client = LLMClient(cfg)
    comments = [CommentInput(id="c1", text="hello world cache probe")]

    raw = {
        "model": "test/model:free",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    content = json.dumps(
        {"items": [{"id": "c1", "label": "neutral", "needs_human": False, "reasons": ["fine"]}]}
    )
    call_count = {"n": 0}

    def fake_chat(model: str, user_content: str, **kwargs: object) -> tuple[str, dict]:
        call_count["n"] += 1
        return content, raw

    with patch.object(client, "_chat", side_effect=fake_chat):
        rows1, _ = client.moderate_batch(comments)
        rows2, _ = client.moderate_batch(comments)
    assert call_count["n"] == 1
    assert rows1[0]["label"].value == "neutral"
    assert rows2[0]["label"].value == "neutral"


def test_provider_data_collection_deny(cfg: OpenRouterConfig) -> None:
    client = LLMClient(cfg)
    extra = client._provider_extra()
    assert extra["provider"]["data_collection"] == "deny"