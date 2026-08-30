"""Health check.

Unauthenticated on purpose: this is the endpoint the keep-warm cron hits every
ten minutes. It performs a trivial database query, which does double duty --
it keeps the Render instance from spinning down (30-60s cold start otherwise)
and keeps the Supabase free-tier project from pausing after 7 days idle.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import Health
from app.config import Settings, get_settings
from app.db.models import Chunk, DocStatus, Document
from app.db.session import get_session

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
async def health(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Health:
    database = "ok"
    approved = 0
    chunks = 0
    try:
        approved = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.status == DocStatus.APPROVED)
                )
            ).scalar_one()
        )
        chunks = int(
            (await session.execute(select(func.count()).select_from(Chunk))).scalar_one()
        )
    except Exception as exc:  # noqa: BLE001 - health must report, not raise
        database = f"error: {type(exc).__name__}"

    return Health(
        status="ok" if database == "ok" else "degraded",
        environment=settings.environment,
        database=database,
        documents_approved=approved,
        chunks_total=chunks,
    )
