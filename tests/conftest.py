"""Shared test configuration.

Unit tests run with no database and no API key. The integration tests in
tests/security/ need a real Postgres with pgvector and skip without one.

Ordering matters here. pydantic-settings gives the process environment
precedence over `.env`, so setting a localhost default before loading `.env`
would silently override a real DATABASE_URL and make the security suite skip
while appearing to pass. Load `.env` first, then fill only what is still
missing.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file() -> None:
    """Minimal .env loader. Existing environment variables win."""
    if not _ENV_FILE.exists():
        return
    for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file()

# Placeholders so the settings object can be constructed for unit tests. These
# only apply when nothing real was supplied.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-unit-tests")  # noqa: S105
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-secret-not-used")  # noqa: S105
os.environ.setdefault("ENVIRONMENT", "development")
