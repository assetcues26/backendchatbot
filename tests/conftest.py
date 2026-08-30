"""Shared test configuration.

Unit tests run with no database and no API key. Integration tests are marked
and skip unless DATABASE_URL points at a real Postgres with pgvector.
"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used-in-unit-tests")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-secret-not-used-in-unit-tests")
os.environ.setdefault("ENVIRONMENT", "development")
