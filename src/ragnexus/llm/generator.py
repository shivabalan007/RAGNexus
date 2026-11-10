from llm.openrouter_client import llm
from llm.prompt import build_prompt


def generate_answer(query, retrieved_chunks, conversation_history=None):

    if not retrieved_chunks:
        return "I don't know based on the provided context."

    # Build the RAG prompt
    prompt = build_prompt(query, retrieved_chunks, conversation_history)

    # Call LLM
    response = llm(prompt, temperature=0.2)

    if response is None:
        return "I couldn't generate a response."

    return response.strip()

"""
Takes query and retrieved texts, builds prompt, calls LLM, returns answer. Main generation function called by all three RAG pipelines.
"""