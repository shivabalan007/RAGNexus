from ragnexus.config import get_settings
from ragnexus.embeddings.query_embedder import get_query_embedder
from ragnexus.llm.providers.factory import build_chat_model
from ragnexus.rag.orchestration.service import RAGOrchestrationService
from ragnexus.retrieval.query_rewriter import QueryRewriter
from ragnexus.retrieval.advanced import MultiQueryGenerator
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.service import RetrievalService
from ragnexus.agentic.graph import AgenticRAGWorkflow
from ragnexus.agentic.tools.tavily_search import TavilySearchTool
from ragnexus.infrastructure.cache import build_cache, CachedEmbeddingFunction

def build_orchestration_service() -> RAGOrchestrationService:
    settings = get_settings()
    chat_model = build_chat_model(settings)
    query_embedder = get_query_embedder()
    cache = build_cache(settings)
    cached_embed_query = CachedEmbeddingFunction(query_embedder.embed_query, model_name=settings.EMBEDDING_MODEL, cache=cache, ttl=settings.REDIS_TTL_SECONDS)
    retrieval = RetrievalService(
        embed_query=cached_embed_query,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        reranker=CrossEncoderReranker(model_name=settings.RERANKER_MODEL),
        multi_query_generator=MultiQueryGenerator(
            generate=lambda prompt: chat_model.complete(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=220
            ).text,
            max_queries=settings.MULTI_QUERY_MAX_QUERIES,
        ),
        query_rewriter=QueryRewriter(
            generate=lambda prompt: chat_model.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            ).text
        ),
    )
    tavily_key = settings.TAVILY_API_KEY.get_secret_value() if settings.TAVILY_API_KEY else None
    tavily = TavilySearchTool(
        tavily_key, max_results=settings.TAVILY_MAX_RESULTS,
        search_depth=settings.TAVILY_SEARCH_DEPTH,
    )
    try:
        agentic = AgenticRAGWorkflow(retrieval_service=retrieval, chat_model=chat_model, tavily=tavily)
    except RuntimeError:
        # Keep Standard RAG available if an optional agentic dependency was not installed.
        agentic = None
    return RAGOrchestrationService(
        retrieval_service=retrieval, chat_model=chat_model, agentic_workflow=agentic
    )
