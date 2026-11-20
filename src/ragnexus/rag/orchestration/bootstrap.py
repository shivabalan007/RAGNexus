from ragnexus.config import get_settings
from ragnexus.embeddings.query_embedder import get_query_embedder
from ragnexus.infrastructure.cache.redis_cache import CachedEmbeddingFunction, build_cache
from ragnexus.llm.providers.factory import build_chat_model
from ragnexus.rag.orchestration.service import RAGOrchestrationService
from ragnexus.retrieval.query_rewriter import QueryRewriter
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.service import RetrievalService


def build_orchestration_service() -> RAGOrchestrationService:
    settings = get_settings()
    chat_model = build_chat_model(settings)
    cache = build_cache(settings)
    query_embedder = get_query_embedder()
    cached_embed_query = CachedEmbeddingFunction(
        query_embedder.embed_query,
        model_name=settings.EMBEDDING_MODEL,
        cache=cache,
        ttl=settings.REDIS_EMBEDDING_TTL_SECONDS,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
    )
    retrieval = RetrievalService(
        embed_query=cached_embed_query,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        reranker=CrossEncoderReranker(model_name=settings.RERANKER_MODEL),
        query_rewriter=QueryRewriter(
            generate=lambda prompt: chat_model.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            ).text
        ),
        cache=cache,
    )
    return RAGOrchestrationService(
        retrieval_service=retrieval,
        chat_model=chat_model,
        history_limit=settings.CONVERSATION_HISTORY_LIMIT,
        intent_router=build_intent_router(chat_model),
        cache=cache,
    )


def build_agentic_query_adapter(chat_model=None):
    """Return the advanced-retrieval query generator backed by the selected provider."""
    from ragnexus.advance_retrieval.integration import LLMQueryAdapter

    model = chat_model or build_chat_model(get_settings())
    return LLMQueryAdapter(
        lambda prompt: model.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        ).text
    )


def build_intent_router(chat_model=None):
    """Build the shared, low-latency conversation intent router."""
    from ragnexus.rag.routing.intent_router import IntentRouter

    return IntentRouter(chat_model or build_chat_model(get_settings()))


def build_advanced_retrieval_engine(
    retriever,
    *,
    config=None,
    parent_expander=None,
    evidence_verifier=None,
    chat_model=None,
):
    """Wire advanced multi-query/decomposition planning to the selected provider."""
    from ragnexus.advance_retrieval.engine import AdvancedRetrievalEngine

    model = chat_model or build_chat_model(get_settings())
    return AdvancedRetrievalEngine(
        retriever=retriever,
        query_llm=build_agentic_query_adapter(model),
        parent_expander=parent_expander,
        evidence_verifier=evidence_verifier,
        config=config,
    )
