from types import SimpleNamespace

from pydantic import SecretStr

from ragnexus.llm.providers.openai_compatible import OpenAICompatibleChatModel


def settings(**overrides):
    values = {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": None,
        "OLLAMA_MODEL": "qwen2.5:7b",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_API_KEY": None,
        "OPENROUTER_API_KEY": None,
        "GROQ_API_KEY": None,
        "LLM_TIMEOUT_SECONDS": 45,
        "LLM_MAX_RETRIES": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ollama_uses_openai_compatible_local_endpoint_without_cloud_key():
    model = OpenAICompatibleChatModel(settings())
    assert model.base_url == "http://localhost:11434/v1"
    assert model.api_key == "ollama"
    assert model.model_name == "qwen2.5:7b"


def test_ollama_allows_generic_model_override():
    model = OpenAICompatibleChatModel(settings(LLM_MODEL="llama3.1:8b"))
    assert model.model_name == "llama3.1:8b"


def test_openrouter_keeps_existing_endpoint_and_key():
    model = OpenAICompatibleChatModel(settings(
        LLM_PROVIDER="openrouter",
        OPENROUTER_API_KEY=SecretStr("test-key"),
        LLM_MODEL="custom/openrouter-model",
    ))
    assert model.base_url == "https://openrouter.ai/api/v1"
    assert model.api_key == "test-key"
    assert model.model_name == "custom/openrouter-model"


def test_provider_retries_transient_server_error(monkeypatch):
    from types import SimpleNamespace
    from ragnexus.llm.providers import openai_compatible

    calls = {"count": 0}

    class TemporaryServerError(Exception):
        status_code = 503

    class FakeCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TemporaryServerError()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    model = OpenAICompatibleChatModel(settings())
    monkeypatch.setattr(model, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(openai_compatible, "sleep", lambda _seconds: None)
    result = model.complete([{"role": "user", "content": "hello"}])
    assert result.text == "ready"
    assert result.retry_count == 1
    assert calls["count"] == 2
