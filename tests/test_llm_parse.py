from __future__ import annotations

import json

import pytest

from engine.llm.client import LLMClient, LLMParseError
from engine.llm.config import OpenRouterConfig
from engine.llm.parse import strip_llm_json_wrapper


@pytest.fixture
def cfg() -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        models=("test/model:free",),
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


def test_strip_think_and_fenced_json() -> None:
    raw = (
        "<think>planning the json</think>\n"
        "```json\n"
        '{"items": [{"id": "a", "label": "neutral", "needs_human": false, "reasons": []}]}\n'
        "```"
    )
    text = strip_llm_json_wrapper(raw)
    data = json.loads(text)
    assert data["items"][0]["id"] == "a"


def test_parse_items_with_think_block(cfg: OpenRouterConfig) -> None:
    client = LLMClient(cfg)
    content = (
        "<think>reasoning</think>\n"
        "```json\n"
        '{"items": [{"id": "x", "label": "toxic", "needs_human": true, "reasons": ["insult"]}]}\n'
        "```"
    )
    items = client._parse_items(content, {"x"})
    assert items[0].label == "toxic"


def test_parse_items_bare_fenced(cfg: OpenRouterConfig) -> None:
    client = LLMClient(cfg)
    content = '```\n{"items": [{"id": "y", "label": "neutral", "needs_human": false, "reasons": []}]}\n```'
    items = client._parse_items(content, {"y"})
    assert items[0].label == "neutral"


def test_parse_id_mismatch(cfg: OpenRouterConfig) -> None:
    client = LLMClient(cfg)
    content = json.dumps(
        {"items": [{"id": "a", "label": "neutral", "needs_human": False, "reasons": []}]}
    )
    with pytest.raises(LLMParseError):
        client._parse_items(content, {"a", "b"})