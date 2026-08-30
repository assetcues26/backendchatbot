"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import admin, chat, health
from app.config import get_settings
from app.db.session import dispose_engine

logger = logging.getLogger("assetcues")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    logger.info(
        "starting environment=%s chat_model=%s embedding_model=%s",
        settings.environment,
        settings.openai_chat_model,
        settings.openai_embedding_model,
    )
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AssetCues RBAC Document Assistant",
        description=(
            "Role-aware question answering over AssetCues product documentation. "
            "Access is enforced in the retrieval query, not in the prompt."
        ),
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs are useful in development and are an information
        # disclosure in production.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api")
    app.include_router(admin.router, prefix="/api")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        """Never leak an internal error message to a caller.

        A stack trace or a database error string can disclose table names,
        document titles, or query structure. The detail goes to the logs; the
        caller gets a generic failure.
        """
        logger.exception("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
