from __future__ import annotations

from collections.abc import Callable


class LazyOrchestrationService:
    """Build the shared RAG orchestration service on first chat request."""

    def __init__(self, builder: Callable[[], object]):
        self._builder = builder
        self._service = None

    def _get_service(self):
        if self._service is None:
            self._service = self._builder()
        return self._service

    def chat(self, *args, **kwargs):
        return self._get_service().chat(*args, **kwargs)

    def agentic_chat(self, *args, **kwargs):
        return self._get_service().agentic_chat(*args, **kwargs)

    def invalidate_session_cache(self, *args, **kwargs):
        # Do not eagerly build the model/retrieval stack for cache cleanup.
        if self._service is not None:
            invalidate = getattr(self._service, "invalidate_session_cache", None)
            if callable(invalidate):
                return invalidate(*args, **kwargs)
        return None
