"""Proxy inference to LMStudio's OpenAI-compatible API."""

from __future__ import annotations

from openai import OpenAI

from app.config import LMSTUDIO_BASE_URL, LLAMA_SYSTEM_PROMPT

_client = None  # type: Optional[OpenAI]


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
    return _client


async def generate(text: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.2-1b-instruct",  # LMStudio model identifier
        messages=[
            {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Rewrite this sarcastic headline as a neutral, non-sarcastic news headline:\n\n{text}",
            },
        ],
        temperature=0,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()


async def check_connection() -> bool:
    try:
        client = _get_client()
        client.models.list()
        return True
    except Exception:
        return False
