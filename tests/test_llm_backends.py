import pytest

from engine.llm.backends import llm_backend, local_llm_model


def test_llm_backend_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "local")
    assert llm_backend() == "local"
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.2:3b")
    assert local_llm_model() == "llama3.2:3b"