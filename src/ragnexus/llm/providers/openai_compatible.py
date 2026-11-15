from __future__ import annotations

from typing import Any

from ragnexus.llm.providers.base import LLMResponse
from ragnexus.observability.tracing import start_span


class ProviderError(RuntimeError):
    """Raised when an LLM provider call fails or returns unusable output."""


class OpenAICompatibleChatModel:
    """Lazy OpenAI-compatible client for OpenRouter or Groq.

    The OpenAI SDK is used as the transport only. Provider credentials are
    read from Settings and are never logged.
    """

    PROVIDERS = {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "key_field": "OPENROUTER_API_KEY",
            "default_model": "openai/gpt-oss-120b:free",
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "key_field": "GROQ_API_KEY",
            "default_model": "openai/gpt-oss-120b",
        },
    }

    def __init__(self, settings: Any):
        self.settings = settings
        self.provider_name = settings.LLM_PROVIDER.lower()
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider_name}")
        provider = self.PROVIDERS[self.provider_name]
        secret = getattr(settings, provider["key_field"], None)
        self.api_key = secret.get_secret_value() if secret else None
        if not self.api_key:
            raise ValueError(f"{provider['key_field']} is not configured")
        self.base_url = provider["base_url"]
        self.model_name = settings.LLM_MODEL or provider["default_model"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError("Install the 'openai' package to use LLM providers") from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=45.0, max_retries=2)
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages cannot be empty")
        with start_span("ragnexus.llm.complete", {
            "ragnexus.llm.provider": self.provider_name,
            "ragnexus.llm.model": self.model_name,
            "ragnexus.llm.max_tokens": max_tokens,
        }) as span:
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if not response.choices:
                    raise ProviderError(
                        f"{self.provider_name} returned no choices. "
                        f"Model={self.model_name}, "
                        f"Usage={response.usage}"
                    )

                choice = response.choices[0]
                message = choice.message
                content = message.content

                if not isinstance(content, str) or not content.strip():
                    has_reasoning = bool(getattr(message, "reasoning", None))
                    has_tool_calls = bool(getattr(message, "tool_calls", None))

                    raise ProviderError(
                        f"{self.provider_name} returned no usable text. "
                        f"Model={self.model_name}, "
                        f"FinishReason={choice.finish_reason}, "
                        f"ContentType={type(content).__name__}, "
                        f"HasReasoning={has_reasoning}, "
                        f"HasToolCalls={has_tool_calls}"
                    )

                text = content
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if span is not None:
                    if prompt_tokens is not None:
                        span.set_attribute("ragnexus.llm.prompt_tokens", int(prompt_tokens))
                    if completion_tokens is not None:
                        span.set_attribute("ragnexus.llm.completion_tokens", int(completion_tokens))
                return LLMResponse(
                    text=text.strip(),
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except ProviderError:
                raise
            except Exception as exc:
                # Do not include credentials, request headers, or raw provider payloads.
                raise ProviderError(f"{self.provider_name} request failed: {type(exc).__name__}") from exc
