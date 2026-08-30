"""Proposes an access classification for a newly ingested document.

This is a *proposal*, never a decision. The output lands in
`documents.suggested_*` and the document stays in PENDING_REVIEW until a human
administrator approves it. That separation is the point: the model is allowed
to do the tedious first pass across hundreds of files, and a person is still
the one who grants access.

If the model call fails for any reason we fall back to the most restrictive
classification available -- RESTRICTED, admin only. A classifier outage must
never open a document up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.principal import ROLE_ADMIN
from app.db.models import Sensitivity
from app.ingest.parsers import ParsedDocument
from app.rag import llm
from app.rag.prompts import CLASSIFICATION_SCHEMA, CLASSIFIER_SYSTEM_PROMPT

# How much of the document the classifier sees. The AssetCues template puts
# purpose, audience and reading rules in the first pages, which is exactly
# what distinguishes a customer guide from an internal specification.
_EXCERPT_CHARS = 6000


@dataclass(frozen=True, slots=True)
class Classification:
    sensitivity: int
    role_keys: list[str]
    doc_type: str
    rationale: str
    failed: bool = False


FAILSAFE = Classification(
    sensitivity=int(Sensitivity.RESTRICTED),
    role_keys=[ROLE_ADMIN],
    doc_type="Document",
    rationale=(
        "Automatic classification failed, so this document defaults to the most "
        "restrictive setting. An administrator must classify it manually."
    ),
    failed=True,
)


def build_classifier_input(
    parsed: ParsedDocument, filename: str, module: str
) -> str:
    audience = ", ".join(parsed.declared_audience) or "(not stated)"
    excerpt = parsed.markdown[:_EXCERPT_CHARS]
    return (
        f"FILENAME: {filename}\n"
        f"MODULE (source folder): {module or '(unknown)'}\n"
        f"TITLE: {parsed.title}\n"
        f"DOCUMENT TYPE (from filename): {parsed.doc_type}\n"
        f"DECLARED PRIMARY AUDIENCE (verbatim from the document): {audience}\n"
        f"APPROX LENGTH: {parsed.word_count} words\n\n"
        f"EXCERPT:\n{excerpt}"
    )


async def classify(
    parsed: ParsedDocument, filename: str, module: str
) -> Classification:
    """Ask the model to propose sensitivity and readers. Never raises."""
    try:
        raw = await llm.complete_json(
            CLASSIFIER_SYSTEM_PROMPT,
            build_classifier_input(parsed, filename, module),
            CLASSIFICATION_SCHEMA,
            "document_classification",
            max_tokens=600,
        )
        data = json.loads(raw)
    except Exception:
        return FAILSAFE

    return _normalise(data, parsed)


def _normalise(data: dict[str, object], parsed: ParsedDocument) -> Classification:
    """Clamp the model's answer into something the database will accept."""
    raw_sensitivity = data.get("sensitivity", Sensitivity.RESTRICTED)
    if isinstance(raw_sensitivity, (int, str)):
        try:
            sensitivity = int(raw_sensitivity)
        except ValueError:
            sensitivity = int(Sensitivity.RESTRICTED)
    else:
        sensitivity = int(Sensitivity.RESTRICTED)
    sensitivity = min(max(sensitivity, 1), 4)

    raw_roles = data.get("role_keys")
    roles = (
        {r.strip().lower() for r in raw_roles if isinstance(r, str)}
        if isinstance(raw_roles, list)
        else set()
    )
    # Admin always retains access; it is the role that fixes mistakes.
    roles.add(ROLE_ADMIN)

    # Consistency guard: a role may not be proposed for a document above its
    # clearance. The database enforces this at query time regardless, but
    # showing an administrator a proposal that could never take effect is
    # confusing, so we filter it here too.
    roles = {r for r in roles if _role_clearance(r) >= sensitivity}
    roles.add(ROLE_ADMIN)

    rationale = str(data.get("rationale") or "").strip()
    doc_type = str(data.get("doc_type") or parsed.doc_type or "Document").strip()

    return Classification(
        sensitivity=sensitivity,
        role_keys=sorted(roles),
        doc_type=doc_type[:100],
        rationale=rationale[:2000],
    )


# Mirrors DEFAULT_ROLES in app/db/seed.py. Kept in sync by
# tests/security/test_role_seed.py.
_ROLE_CLEARANCE = {
    "admin": 4,
    "product": 4,
    "engineering": 4,
    "sales": 4,
    "qa": 3,
    "support": 3,
    "customer": 2,
}


def _role_clearance(role_key: str) -> int:
    return _ROLE_CLEARANCE.get(role_key, 0)
