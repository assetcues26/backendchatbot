"""Per-user rate limiting.

Keeps one account from burning the OpenAI budget, and blunts the "probe the
assistant with a thousand phrasings until something leaks" approach.

In-memory sliding window, which is correct for the single-instance free-tier
deployment. A multi-instance deployment needs Redis -- see docs/RUNBOOK.md.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, status

from app.config import get_settings
from app.core.principal import Principal

_WINDOW_SECONDS = 60
_hits: dict[str, deque[float]] = defaultdict(deque)


async def enforce_rate_limit(principal: Principal) -> None:
    settings = get_settings()
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    now = time.time()
    bucket = _hits[str(principal.user_id)]
    while bucket and now - bucket[0] > _WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= limit:
        retry_after = int(_WINDOW_SECONDS - (now - bucket[0])) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {limit} questions per minute exceeded.",
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)


def reset() -> None:
    """Test hook."""
    _hits.clear()
