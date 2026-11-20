import pytest
from pydantic import ValidationError

from ragnexus.config import Settings


def _settings(**overrides):
    values = {
        "_env_file": None,
        "JWT_SECRET_KEY": "test-secret",
        "DUMMY_PASSWORD_HASH": "test-hash",
        "DATABASE_URL": "postgresql+psycopg2://user:password@localhost:5432/ragnexus",
    }
    values.update(overrides)
    return Settings(**values)


def test_all_runtime_retrieval_and_agentic_settings_are_declared():
    settings = _settings()
    assert settings.MULTI_QUERY_MAX_QUERIES == 3
    assert settings.ENABLE_MULTI_QUERY is True
    assert settings.ENABLE_PARENT_EXPANSION is True
    assert settings.ENABLE_CONTEXTUAL_COMPRESSION is True
    assert settings.ENABLE_EVIDENCE_SUFFICIENCY is True
    assert settings.AGENTIC_MAX_RETRIES == 2
    assert settings.AGENTIC_ENABLE_QUERY_ROUTING is True
    assert settings.AGENTIC_ENABLE_RETRIEVAL_GRADING is True
    assert settings.AGENTIC_ENABLE_ANSWER_VERIFICATION is True
    assert settings.REDIS_ANSWER_TTL_SECONDS == 300
    assert settings.REDIS_FRESH_ANSWER_TTL_SECONDS == 60
    assert settings.REDIS_FRESH_TAVILY_TTL_SECONDS == 60


def test_retrieval_service_constructs_with_declared_settings():
    pytest.importorskip("pgvector.sqlalchemy")
    from ragnexus.retrieval.service import RetrievalService

    service = RetrievalService(embed_query=lambda _query: [0.0] * 384, settings=_settings())

    assert service.settings.MULTI_QUERY_MAX_QUERIES == 3
    assert service.multi_query_generator is not None
    assert service.sufficiency_checker is not None


def test_fixed_pgvector_dimension_is_validated_at_startup():
    with pytest.raises(ValidationError, match="EMBEDDING_DIMENSION must be 384"):
        _settings(EMBEDDING_DIMENSION=768)


def test_lazy_api_orchestrator_delegates_agentic_chat(monkeypatch):
    from ragnexus.api.v1.orchestration_proxy import LazyOrchestrationService

    calls = []
    expected = object()

    class FakeService:
        def agentic_chat(self, *args, **kwargs):
            calls.append((args, kwargs))
            return expected

    lazy = LazyOrchestrationService(lambda: FakeService())

    result = lazy.agentic_chat("db", user_id="user", session_id="session", query="test")

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] == ("db",)
    assert calls[0][1]["query"] == "test"


def test_standard_and_agentic_use_one_cached_orchestration_instance(monkeypatch):
    from ragnexus.api.v1.orchestration_proxy import LazyOrchestrationService

    built = []

    class FakeService:
        def chat(self, *args, **kwargs):
            return "standard"

        def agentic_chat(self, *args, **kwargs):
            return "agentic"

    def build():
        instance = FakeService()
        built.append(instance)
        return instance

    lazy = LazyOrchestrationService(build)

    assert lazy.chat() == "standard"
    assert lazy.agentic_chat() == "agentic"
    assert len(built) == 1
