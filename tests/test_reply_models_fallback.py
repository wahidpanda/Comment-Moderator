from engine.llm.client import LLMClient
from engine.llm.config import OpenRouterConfig, resolve_openrouter_models


def test_resolve_models_when_env_empty(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_SUMMARY_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_DEFAULT_MODEL", "test/model")
    cfg = OpenRouterConfig(
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        models=(),
        batch_size=15,
        prompt_version="moderation-v1",
        max_requests_per_day=10,
        daily_request_buffer=2,
        data_collection="deny",
        zdr=False,
        requests_per_minute=20,
        http_referer=None,
        app_name="pytest",
        log_key_response_once=False,
    )
    assert resolve_openrouter_models(cfg) == ("test/model",)
    client = LLMClient(cfg)
    assert client._resolved_models() == ("test/model",)