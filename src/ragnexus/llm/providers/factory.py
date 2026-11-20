from ragnexus.config import Settings, get_settings
from ragnexus.llm.providers.openai_compatible import OpenAICompatibleChatModel


def build_chat_model(settings: Settings | None = None) -> OpenAICompatibleChatModel:
    """Build the configured LLM provider once for all RAG components."""
    return OpenAICompatibleChatModel(settings or get_settings())
