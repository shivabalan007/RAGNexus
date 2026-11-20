from __future__ import annotations

from typing import Any
from time import sleep

from ragnexus.llm.providers.base import LLMResponse


class ProviderError(RuntimeError):
    """Raised when an LLM provider call fails or returns unusable output."""


class OpenAICompatibleChatModel:
    """OpenAI-compatible client for OpenRouter, Ollama, and Groq.

    OpenRouter's existing endpoint/model defaults are preserved. Ollama uses
    its OpenAI-compatible /v1 endpoint and a local model; no cloud API key is
    required (the SDK requires a non-empty placeholder key).
    """

    PROVIDERS = {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "key_field": "OPENROUTER_API_KEY",
            "default_model": "openai/gpt-oss-120b:free",
            "requires_key": True,
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "key_field": "GROQ_API_KEY",
            "default_model": "openai/gpt-oss-120b",
            "requires_key": True,
        },
        "ollama": {
            "key_field": "OLLAMA_API_KEY",
            "default_model": "llama3.2",
            "requires_key": False,
        },
    }

    def __init__(self, settings: Any):
        self.settings = settings
        self.provider_name = str(settings.LLM_PROVIDER).lower()
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider_name}")

        provider = self.PROVIDERS[self.provider_name]
        secret = getattr(settings, provider["key_field"], None)
        configured_key = secret.get_secret_value() if secret else None

        if provider["requires_key"] and not configured_key:
            raise ValueError(f"{provider['key_field']} is not configured")

        if self.provider_name == "ollama":
            self.base_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"
            self.api_key = configured_key or "ollama"
            default_model = settings.OLLAMA_MODEL
        else:
            self.base_url = provider["base_url"]
            self.api_key = configured_key
            default_model = provider["default_model"]

        self.model_name = settings.LLM_MODEL or default_model
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(
                    "Install the 'openai' package to use OpenAI-compatible LLM providers"
                ) from exc
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        return status is None or status == 429 or (isinstance(status, int) and status >= 500)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages cannot be empty")

        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not response.choices:
                    raise ProviderError(f"{self.provider_name} returned no choices")
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ProviderError(f"{self.provider_name} returned an empty response")
                usage = response.usage
                return LLMResponse(
                    text=content.strip(),
                    model=self.model_name,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    retry_count=attempt,
                )
            except ProviderError:
                raise
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    # Never include credentials, request headers, or raw provider payloads.
                    raise ProviderError(
                        f"{self.provider_name} request failed after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}"
                    ) from exc
                sleep(min(0.25 * (2 ** attempt), 2.0))

        raise ProviderError(f"{self.provider_name} request failed")
