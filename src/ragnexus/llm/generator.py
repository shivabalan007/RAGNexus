from __future__ import annotations

from ragnexus.llm.providers.base import ChatModel
from ragnexus.llm.providers.factory import build_chat_model
from ragnexus.llm.prompt import build_prompt


def generate_answer(
    query,
    retrieved_chunks,
    conversation_history=None,
    *,
    chat_model: ChatModel | None = None,
):
    """Generate a grounded answer through the configured provider."""
    if not retrieved_chunks:
        return "I don't know based on the provided context."

    prompt = build_prompt(query, retrieved_chunks, conversation_history)
    model = chat_model or build_chat_model()
    try:
        response = model.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception:
        return "I couldn't generate a response."

    return response.text.strip() if response and response.text else "I couldn't generate a response."
