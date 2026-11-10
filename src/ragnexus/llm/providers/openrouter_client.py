import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

DEFAULT_MAX_TOKENS = 800

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable is not configured."
    )

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=api_key,
)

def _is_invalid_response(response: str | None) -> bool:
    """
    Detect unusable LLM responses.

    Returns True when the model returns:
    - None
    - empty output
    - whitespace only
    - padding tokens such as <pad>
    """

    if response is None:
        return True

    response = str(response).strip()

    if not response:
        return True

    # Detect repeated padding tokens returned by some models.
    tokens = response.split()

    if tokens and all(token == "<pad>" for token in tokens):
        return True

    # Also reject responses containing excessive padding.
    pad_count = response.count("<pad>")

    if pad_count >= 3:
        return True

    return False

def llm(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
):
    """
    Call the configured OpenRouter model.

    Returns:
        str  -> valid model response
        None -> unusable/invalid model response

    The caller is responsible for applying an appropriate
    fallback when None is returned.
    """

    if not prompt or not str(prompt).strip():
        return None

    try:

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    except Exception as exc:

        print("\n========== LLM ERROR ==========")
        print(type(exc).__name__, ":", exc)
        print("=" * 50)

        return None

    if not completion:
        return None

    if not completion.choices:
        return None

    message = completion.choices[0].message

    if message is None:
        return None

    response = message.content

    if _is_invalid_response(response):

        print("\n========== INVALID LLM RESPONSE ==========")
        print("Model    :", model)
        print("Response :", repr(response))
        print("=" * 50)

        return None

    return str(response).strip()

    # model="deepseek/deepseek-chat",
    # mistralai/mistral-small-3.1-24b-instruct
    # meta-llama/llama-3-8b-instruct
    # meta-llama/llama-3-70b-instruct
    # mistralai/mistral-large

"""
OpenAI-compatible client pointed at OpenRouter API with DeepSeek model. Single llm(prompt, temperature) function used by all evaluators and generator.
"""

"""
google/gemma-4-26b-a4b-it:free
openai/gpt-oss-120b:free
nvidia/nemotron-3-ultra:free
nvidia/nemotron-3-super:free
openai/gpt-oss-20b:free
nvidia/nemotron-3-nano-30b-a3b:free

"""