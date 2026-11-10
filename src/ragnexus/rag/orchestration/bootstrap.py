from ragnexus.config import get_settings
from ragnexus.embeddings.query_embedder import get_query_embedder
from ragnexus.llm.providers.factory import build_chat_model
from ragnexus.rag.orchestration.service import RAGOrchestrationService
from ragnexus.retrieval.query_rewriter import QueryRewriter
from ragnexus.retrieval.reranker import CrossEncoderReranker
from ragnexus.retrieval.service import RetrievalService

def build_orchestration_service() -> RAGOrchestrationService:
    settings = get_settings()
    chat_model = build_chat_model(settings)
    query_embedder = get_query_embedder()
    retrieval = RetrievalService(
        embed_query=query_embedder.embed_query,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        reranker=CrossEncoderReranker(),
        query_rewriter=QueryRewriter(
            generate=lambda prompt: chat_model.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            ).text
        ),
    )
    return RAGOrchestrationService(retrieval_service=retrieval, chat_model=chat_model)
