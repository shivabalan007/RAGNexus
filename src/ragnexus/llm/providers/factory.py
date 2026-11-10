from ragnexus.config import Settings, get_settings
from ragnexus.llm.providers.openai_compatible import OpenAICompatibleChatModel


def build_chat_model(settings: Settings | None = None) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(settings or get_settings())
