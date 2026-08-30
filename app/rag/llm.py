"""OpenAI client wrappers: chat, streaming chat, and embeddings.

Deliberately thin. The provider is isolated behind these three functions so
that swapping models -- the most likely tuning change -- is a config edit, and
so the chat model and the embedding model can move independently.

Nothing here makes an access-control decision. By the time text reaches this
module the ACL has already been enforced in SQL (G3) and re-verified (G4).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from openai import APIError, AsyncOpenAI, RateLimitError

from app.config import get_settings

# Embedding inputs are batched; OpenAI accepts large batches but we keep them
# modest so a single failure retries cheaply.
_EMBED_BATCH_SIZE = 64
_MAX_RETRIES = 4


@lru_cache
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env (local) or to the "
            "service environment variables (Render/Vercel)."
        )
    return AsyncOpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=0)


async def _with_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Exponential backoff for rate limits and transient API errors."""
    delay = 1.0
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except (RateLimitError, APIError) as exc:
            last = exc
            if attempt == _MAX_RETRIES - 1:
                break
            await asyncio.sleep(delay)
            delay *= 2
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, preserving order.

    Asserts the returned dimension matches the configured one. A silent
    mismatch between the model and the `vector(N)` column would otherwise
    surface much later as an opaque database error.
    """
    if not texts:
        return []

    settings = get_settings()
    client = get_client()
    out: list[list[float]] = []

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = await _with_retry(
            client.embeddings.create,
            model=settings.openai_embedding_model,
            input=batch,
        )
        # The API returns items in input order, but it also carries an explicit
        # index. Sort by it rather than trusting position.
        for item in sorted(response.data, key=lambda d: d.index):
            vec = list(item.embedding)
            if len(vec) != settings.embedding_dim:
                raise RuntimeError(
                    f"Embedding model {settings.openai_embedding_model} returned "
                    f"{len(vec)} dimensions but EMBEDDING_DIM is "
                    f"{settings.embedding_dim}. Update EMBEDDING_DIM and the "
                    f"chunks.embedding column together, then re-embed."
                )
            out.append(vec)

    return out


async def embed_query(query: str) -> list[float]:
    vectors = await embed_texts([query])
    return vectors[0]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 1500,
) -> str:
    """Single non-streaming completion. Used by the classifier."""
    settings = get_settings()
    client = get_client()
    response = await _with_retry(
        client.chat.completions.create,
        model=model or settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def complete_json(
    system: str,
    user: str,
    schema: dict[str, Any],
    schema_name: str,
    *,
    model: str | None = None,
    max_tokens: int = 1500,
) -> str:
    """Completion constrained to a JSON schema.

    Used by the document classifier, where a malformed response would leave a
    document stuck in PROCESSING. Strict mode guarantees parseable output.
    """
    settings = get_settings()
    client = get_client()
    response = await _with_retry(
        client.chat.completions.create,
        model=model or settings.openai_classifier_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or "{}"


async def stream_completion(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 1500,
) -> AsyncIterator[str]:
    """Token stream for the chat endpoint."""
    settings = get_settings()
    client = get_client()
    stream = await client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
        stream=True,
    )
    async for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content
