"""Application settings. Every secret arrives via the environment — never code."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database
    database_url: str = Field(alias="DATABASE_URL")

    # --- OpenAI (single vendor: chat + embeddings)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-5.6-luna", alias="OPENAI_CHAT_MODEL")
    openai_classifier_model: str = Field(
        default="gpt-5.6-luna", alias="OPENAI_CLASSIFIER_MODEL"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    # Asserted against the first live embedding response at startup of any
    # ingest run, so the DB schema can never silently disagree with the model.
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    # --- Supabase Auth
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_jwt_secret: str = Field(default="", alias="SUPABASE_JWT_SECRET")
    supabase_service_role_key: str = Field(
        default="", alias="SUPABASE_SERVICE_ROLE_KEY"
    )

    # --- App
    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")
    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE")

    # --- Retrieval
    retrieval_candidates: int = Field(default=50, alias="RETRIEVAL_CANDIDATES")
    retrieval_top_k: int = Field(default=12, alias="RETRIEVAL_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver: "
                "postgresql+asyncpg://user:pass@host:port/db"
            )
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
