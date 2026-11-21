from __future__ import annotations

from ragnexus.agentic.tools.tavily_search import TavilySearchTool
from ragnexus.infrastructure.cache.redis_cache import CachedEmbeddingFunction, NoopCache
from ragnexus.rag.routing.intent_router import IntentRouter
from ragnexus.rag.routing.router import QueryRouter


class MemoryCache:
    enabled = True

    def __init__(self):
        self.values = {}
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ttl=300):
        self.values[key] = value
        self.ttls[key] = ttl
        return True

    def delete(self, key):
        self.values.pop(key, None)
        self.ttls.pop(key, None)
        return True


def test_intent_router_distinguishes_general_memory_and_knowledge():
    router = IntentRouter()
    assert router.classify("Hi!") == IntentRouter.GENERAL
    assert router.classify("What did we discuss earlier?") == IntentRouter.MEMORY
    assert router.classify("What about the second ROS 2 option?") == IntentRouter.KNOWLEDGE


def test_query_router_requires_user_opt_in_before_web_route():
    router = QueryRouter(enabled=False)
    decision = router.route_query("What is the latest ROS 2 release?", web_enabled=False)
    assert decision.route == "documents"
    assert decision.reason == "web_search_not_enabled"


def test_query_router_routes_freshness_to_hybrid_and_explicit_web_requests_to_web_only():
    router = QueryRouter(enabled=False)
    assert router.route_query("What is the latest ROS 2 release?", web_enabled=True).route == "hybrid"
    explicit = router.route_query("Search the web for ROS 2 release notes", web_enabled=True)
    assert explicit.route == "web"
    assert explicit.reason == "user_explicitly_requested_web_only"


def test_noop_cache_is_explicitly_disabled():
    cache = NoopCache()
    assert cache.enabled is False
    assert cache.get("missing") is None
    assert cache.set("key", {"value": 1}) is False
    assert cache.delete("key") is False


def test_embedding_cache_stores_json_native_vector_and_reuses_it():
    cache = MemoryCache()
    calls = []

    class FakeArray:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    def embed(_query):
        calls.append(_query)
        return FakeArray()

    wrapped = CachedEmbeddingFunction(
        embed, model_name="test-embedding", cache=cache, ttl=180
    )
    assert wrapped("RAG systems") == [0.1, 0.2, 0.3]
    assert wrapped("  RAG   systems ") == [0.1, 0.2, 0.3]
    assert calls == ["RAG systems"]
    # Case-sensitive identifiers must not share a query-embedding cache entry.
    wrapped("rag systems")
    assert calls == ["RAG systems", "rag systems"]


def test_tavily_results_are_cached_and_fresh_queries_use_short_ttl():
    cache = MemoryCache()
    tool = TavilySearchTool(
        "test-key", cache=cache, cache_ttl_seconds=900,
        fresh_cache_ttl_seconds=45,
    )
    calls = []

    class FakeClient:
        def search(self, **_kwargs):
            calls.append(1)
            return {"results": [{
                "title": "ROS release",
                "url": "https://example.com/ros-release",
                "content": "Release announcement",
                "score": 0.9,
            }]}

    tool._client = FakeClient()
    first, first_info = tool.search_with_metadata("latest ROS 2 release")
    second, second_info = tool.search_with_metadata(" latest   ROS 2 release ")
    assert first == second
    assert first_info["cache_hit"] is False
    assert second_info["cache_hit"] is True
    assert len(calls) == 1
    assert next(iter(cache.ttls.values())) == 45


def test_answer_cache_key_reuses_only_same_session_same_pipeline_same_context(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4
    import pytest

    pytest.importorskip("pgvector.sqlalchemy")

    import ragnexus.rag.orchestration.service as orchestration_module
    from ragnexus.rag.orchestration.service import RAGOrchestrationService

    settings = SimpleNamespace(
        REDIS_CONVERSATION_TTL_SECONDS=1800,
        REDIS_ANSWER_TTL_SECONDS=300,
        REDIS_FRESH_ANSWER_TTL_SECONDS=60,
        REDIS_FRESH_TAVILY_TTL_SECONDS=60,
        LLM_PROVIDER="openrouter",
        LLM_MODEL="test-model",
        OLLAMA_MODEL="llama3.2",
        EMBEDDING_MODEL="test-embedding",
        RERANKER_MODEL="test-reranker",
        ENABLE_MULTI_QUERY=True,
        ENABLE_PARENT_EXPANSION=True,
        ENABLE_CONTEXTUAL_COMPRESSION=True,
        MULTI_QUERY_MAX_QUERIES=3,
        EVIDENCE_MIN_TOKEN_OVERLAP=0.3,
        EVIDENCE_MIN_DENSE_SIMILARITY=0.4,
        AGENTIC_MAX_RETRIES=2,
        AGENTIC_ENABLE_RETRIEVAL_GRADING=True,
        AGENTIC_ENABLE_ANSWER_VERIFICATION=True,
        TAVILY_SEARCH_DEPTH="basic",
        TAVILY_MAX_RESULTS=5,
    )
    monkeypatch.setattr(orchestration_module, "get_settings", lambda: settings)

    cache = MemoryCache()

    class FakeRetrieval:
        fingerprint = "corpus-v1"

        def corpus_fingerprint(self, _db, _workspace_id, _document_ids):
            return self.fingerprint

    service = RAGOrchestrationService(
        retrieval_service=FakeRetrieval(),
        chat_model=type("Model", (), {"model_name": "test-model"})(),
        cache=cache,
    )
    session = SimpleNamespace(
        id=uuid4(), workspace_id=uuid4(), visibility="private", document_scope="workspace"
    )
    user_id = uuid4()
    query = "Explain hybrid retrieval"
    prior_history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "Retrieval augmented generation."},
    ]

    original = service._answer_cache_key(
        object(), chat_session=session, user_id=user_id, query=query,
        history=prior_history, pipeline_mode="standard", document_ids=None,
        candidate_k=20, final_k=5,
    )
    repeated_history = [
        *prior_history,
        {"role": "user", "content": query},
        {"role": "assistant", "content": "Hybrid retrieval combines dense and sparse search."},
    ]
    repeated = service._answer_cache_key(
        object(), chat_session=session, user_id=user_id, query=query,
        history=repeated_history, pipeline_mode="standard", document_ids=None,
        candidate_k=20, final_k=5,
    )
    agentic = service._answer_cache_key(
        object(), chat_session=session, user_id=user_id, query=query,
        history=repeated_history, pipeline_mode="agentic", document_ids=None,
        candidate_k=20, final_k=5,
    )
    assert original == repeated
    assert original != agentic

    service.retrieval_service.fingerprint = "corpus-v2"
    after_reindex = service._answer_cache_key(
        object(), chat_session=session, user_id=user_id, query=query,
        history=repeated_history, pipeline_mode="standard", document_ids=None,
        candidate_k=20, final_k=5,
    )
    assert original != after_reindex


def test_agentic_hybrid_route_searches_web_even_when_documents_are_sufficient():
    from types import SimpleNamespace

    from ragnexus.agentic.graph import AgenticRAGWorkflow

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.settings = SimpleNamespace(AGENTIC_MAX_RETRIES=2)
    action = workflow._after_grade({
        "route": "hybrid",
        "retrieval_sufficient": True,
        "retry_count": 0,
    })
    assert action == "web"


def test_agentic_uses_web_as_fallback_after_document_retries_are_exhausted():
    from types import SimpleNamespace

    from ragnexus.agentic.graph import AgenticRAGWorkflow

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.settings = SimpleNamespace(AGENTIC_MAX_RETRIES=2)
    action = workflow._after_grade({
        "route": "documents",
        "retrieval_sufficient": False,
        "retry_count": 2,
        "enable_web_search": True,
    })
    assert action == "web"


def test_agentic_does_not_use_web_when_user_opt_in_is_disabled():
    from types import SimpleNamespace

    from ragnexus.agentic.graph import AgenticRAGWorkflow

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.settings = SimpleNamespace(AGENTIC_MAX_RETRIES=0)
    action = workflow._after_grade({
        "route": "documents",
        "retrieval_sufficient": False,
        "retry_count": 0,
        "enable_web_search": False,
    })
    assert action == "generate"


def test_agentic_node_timer_accumulates_retries():
    from ragnexus.agentic.graph import AgenticRAGWorkflow

    wrapped = AgenticRAGWorkflow._timed_node("retrieval", lambda _state: {"retrieval_result": True})
    first = wrapped({"stage_timings_ms": {"retrieval": 2.0}})
    assert first["retrieval_result"] is True
    assert first["stage_timings_ms"]["retrieval"] >= 2.0


def test_agentic_generation_can_answer_from_web_only_evidence():
    from ragnexus.agentic.graph import AgenticRAGWorkflow
    from ragnexus.llm.providers.base import LLMResponse
    from ragnexus.retrieval.result import RetrievalResult

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            text = (
                "The release was announced in the official notes [1]."
                if self.calls == 1
                else '{"items":[{"segment":1,"source_numbers":[1]}]}'
            )
            return LLMResponse(text=text, model="test-model", prompt_tokens=100, completion_tokens=20)

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.chat_model = FakeModel()
    update = workflow._generate({
        "original_query": "What is the latest release?",
        "retrieval": RetrievalResult(query="What is the latest release?"),
        "retrieval_sufficient": False,
        "web_sources": [{
            "title": "Official release notes",
            "url": "https://example.com/release",
            "content": "The release was announced.",
        }],
        "history": [],
    })
    assert update["generation_attempted"] is True
    assert update["abstained"] is False
    assert update["citations"][0]["source_type"] == "web"
    assert update["generation_latency_ms"] is not None


def test_followup_resolver_skips_model_call_for_standalone_question():
    from ragnexus.rag.orchestration.followup import FollowUpResolver

    calls = []

    class FakeModel:
        def complete(self, *_args, **_kwargs):
            calls.append(1)
            raise AssertionError("standalone questions should not be rewritten")

    resolver = FollowUpResolver(FakeModel())
    history = [{"role": "user", "content": "We discussed ROS 2 navigation."}]
    assert resolver.resolve("Explain hybrid retrieval", history) == "Explain hybrid retrieval"
    assert calls == []


def test_followup_resolver_uses_history_for_referential_question():
    from ragnexus.llm.providers.base import LLMResponse
    from ragnexus.rag.orchestration.followup import FollowUpResolver

    calls = []

    class FakeModel:
        def complete(self, *_args, **_kwargs):
            calls.append(1)
            return LLMResponse("Explain the second navigation approach", "test-model")

    resolver = FollowUpResolver(FakeModel())
    history = [{"role": "assistant", "content": "Option one is SLAM; option two is visual odometry."}]
    assert resolver.resolve("Explain the second one", history) == "Explain the second navigation approach"
    assert calls == [1]


def test_agentic_generation_abstention_triggers_enabled_web_fallback_once():
    from ragnexus.agentic.graph import AgenticRAGWorkflow

    assert AgenticRAGWorkflow._after_generate({
        "enable_web_search": True,
        "web_search_attempted": False,
        "abstained": True,
    }) == "web"
    assert AgenticRAGWorkflow._after_generate({
        "enable_web_search": True,
        "web_search_attempted": True,
        "abstained": True,
    }) == "verify"
    assert AgenticRAGWorkflow._after_generate({
        "enable_web_search": False,
        "web_search_attempted": False,
        "abstained": True,
    }) == "verify"
    assert AgenticRAGWorkflow._after_generate({
        "enable_web_search": True,
        "web_search_attempted": False,
        "abstained": False,
    }) == "verify"


def test_tavily_accepts_pydantic_secret_string():
    from pydantic import SecretStr

    tool = TavilySearchTool(SecretStr("test-key"))
    assert tool.available is True
    assert tool.api_key == "test-key"


def test_document_list_endpoint_exposes_indexed_chunk_and_embedding_fields():
    from types import SimpleNamespace
    from uuid import uuid4
    import pytest

    pytest.importorskip("pgvector.sqlalchemy")
    from ragnexus.api.v1.endpoints.ingestion import list_documents

    workspace_id = uuid4()
    user_id = uuid4()
    document_id = uuid4()
    document = SimpleNamespace(
        id=document_id,
        filename="architecture.pdf",
        status="indexed",
        content_hash="sha256:test",
        document_metadata={"version_number": 3, "is_current": True},
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        file_size=123456,
        mime_type="application/pdf",
        created_at=None,
    )

    class ScalarResult:
        def all(self):
            return [document]

    class ExecuteResult:
        def all(self):
            return [(document_id, 248)]

    class FakeDB:
        def get(self, _model, _key):
            return SimpleNamespace(role="owner")

        def scalars(self, _statement):
            return ScalarResult()

        def execute(self, _statement):
            return ExecuteResult()

    result = list_documents(workspace_id, FakeDB(), SimpleNamespace(id=user_id))
    assert result[0]["indexed_chunk_count"] == 248
    assert result[0]["embedding_model"] == "all-MiniLM-L6-v2"
    assert result[0]["embedding_dimension"] == 384
    assert result[0]["file_size_bytes"] == 123456



def test_agentic_explicit_web_route_searches_web_even_when_documents_are_sufficient():
    from types import SimpleNamespace

    from ragnexus.agentic.graph import AgenticRAGWorkflow

    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.settings = SimpleNamespace(AGENTIC_MAX_RETRIES=2)
    action = workflow._after_grade({
        "route": "web",
        "retrieval_sufficient": True,
        "retry_count": 0,
    })
    assert action == "web"


def test_agentic_web_only_route_does_not_cite_retrieved_documents():
    from ragnexus.agentic.graph import AgenticRAGWorkflow
    from ragnexus.llm.providers.base import LLMResponse
    from ragnexus.retrieval.result import RetrievalResult, RetrievedChunk
    from types import SimpleNamespace
    from uuid import uuid4

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **_kwargs):
            self.calls += 1
            prompt = messages[-1]["content"]
            assert "Private internal document" not in prompt
            if self.calls == 1:
                assert "[SOURCE 1]" in prompt
                text = "The public capital city is Bengaluru [1]."
            else:
                text = '{"items":[{"segment":1,"source_numbers":[1]}]}'
            return LLMResponse(text=text, model="test-model", prompt_tokens=50, completion_tokens=10)

    private_chunk = SimpleNamespace(
        id=uuid4(), document_id=uuid4(), chunk_index=0,
        content="Private internal document says the capital is another city.",
        chunk_metadata={}, document=SimpleNamespace(filename="private.txt", document_metadata={}),
    )
    workflow = object.__new__(AgenticRAGWorkflow)
    workflow.chat_model = FakeModel()
    update = workflow._generate({
        "route": "web",
        "original_query": "Search the web for the capital of Karnataka",
        "retrieval": RetrievalResult(
            query="question", chunks=[RetrievedChunk(chunk=private_chunk)]
        ),
        "retrieval_sufficient": True,
        "web_sources": [{
            "title": "Public reference", "url": "https://example.com/karnataka",
            "content": "Bengaluru is the capital city of Karnataka.",
        }],
        "history": [],
    })
    assert update["abstained"] is False
    assert len(update["citations"]) == 1
    assert update["citations"][0]["source_type"] == "web"



def test_explicit_web_route_skips_document_retrieval_node():
    from ragnexus.agentic.graph import AgenticRAGWorkflow

    assert AgenticRAGWorkflow._after_route({"route": "web"}) == "web"
    assert AgenticRAGWorkflow._after_route({"route": "hybrid"}) == "retrieve"
    assert AgenticRAGWorkflow._after_route({"route": "documents"}) == "retrieve"
