import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_MODEL = "openai/gpt-oss-120b"

DEFAULT_MAX_TOKENS = 800

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not configured in the environment."
    )

client = OpenAI(
    base_url=GROQ_BASE_URL,
    api_key=api_key,
)

def llm(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = GROQ_MODEL,
):


    if not prompt or not prompt.strip():
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

    if response is None:
        return None

    response = response.strip()

    if not response:
        return None

    return response



"""
Groq LLM client for interacting with the Groq API.
"""

"""
openai/gpt-oss-120b
openai/gpt-oss-20b
qwen/qwen3.6-27b

groq/compound
groq/compound-mini
"""