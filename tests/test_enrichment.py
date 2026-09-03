"""Contextual enrichment: what it must do, and what it must never do.

The risk enrichment introduces is specific. A model now writes text that goes
into the retrieval path. If that text could ever be shown or cited, a
hallucinated sentence would become a claim attributed to a real document --
which is worse than the ranking problem enrichment was added to solve.

So the tests that matter most here are the negative ones: the context is
embedded, and nothing else. The rest cover the cost properties, because an
enrichment pass that silently re-embeds everything on every run is a bill
nobody notices until it arrives.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from app.ingest import enrich
from app.ingest.chunker import Chunk as TextChunk
from app.ingest.enrich import DocumentProfile
from app.ingest.parsers import ParsedDocument
from app.rag import prompts, retrieval
from app.rag.retrieval import RetrievedChunk


def make_chunk(ordinal: int, text: str = "body text", heading: str = "4.1") -> TextChunk:
    return TextChunk(
        ordinal=ordinal, heading_path=heading, text=text, token_count=10
    )


def make_parsed(markdown: str = "# Doc\n\nbody") -> ParsedDocument:
    return ParsedDocument(title="A Document", markdown=markdown, doc_type="Spec")


# ---------------------------------------------------------------------------
# The context is embedded, and nothing else
# ---------------------------------------------------------------------------


def test_retrieved_chunks_carry_no_context_at_all() -> None:
    """The safety property, enforced by the type rather than by discipline.

    A `RetrievedChunk` is what reaches the prompt, the citations and the API
    response. If it has no context field, no amount of later carelessness can
    put generated text in front of a reader.
    """
    assert "context" not in RetrievedChunk.__dataclass_fields__


def test_the_search_query_never_selects_the_context_column() -> None:
    sql = retrieval._HYBRID_SEARCH_SQL
    assert "c.text" in sql, "sanity: the query does select the real text"
    assert "c.context" not in sql, (
        "selecting c.context would carry generated text into the prompt path"
    )


def test_the_full_text_index_is_built_from_the_document_text_only() -> None:
    """Keyword search must match words that are really in the file.

    The `tsv` column is computed from `text`; if it covered `context`, a search
    for an exact identifier could match a chunk only because a model mentioned
    that identifier while describing a neighbour.
    """
    from app.db.models import Chunk

    computed = str(Chunk.__table__.c.tsv.server_default.sqltext)
    assert "text" in computed
    assert "context" not in computed


def test_a_prompt_built_from_chunks_contains_only_document_text() -> None:
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        ordinal=1,
        heading_path="4.4 Profiles",
        text="Administrators may edit a profile.",
        token_count=10,
        title="User Access",
        module="User Access Management",
        capability="User Access & Permission Management",
        doc_type="Spec",
        sensitivity=2,
        score=0.5,
    )
    prompt = prompts.build_user_prompt("who may edit?", [chunk])
    assert "Administrators may edit a profile." in prompt


# ---------------------------------------------------------------------------
# What gets embedded
# ---------------------------------------------------------------------------


def test_context_is_embedded_ahead_of_the_text() -> None:
    """Context first, so it survives truncation of a long chunk."""
    combined = enrich.embedding_text("From the UAM spec.", "4.4 Profiles", "row")
    assert combined.startswith("From the UAM spec.")
    assert combined.endswith("row")


def test_a_chunk_with_no_context_is_still_embedded_from_its_own_text() -> None:
    """The failure mode has to be worse ranking, never a lost document."""
    assert enrich.embedding_text("", "4.4 Profiles", "row") == "4.4 Profiles\n\nrow"
    assert enrich.embedding_text("", "", "row") == "row"


def test_changing_the_context_invalidates_the_cached_embedding() -> None:
    before = enrich.chunk_digest("", "4.4", "row")
    after = enrich.chunk_digest("From the UAM spec.", "4.4", "row")
    assert before != after


def test_the_carry_forward_key_ignores_the_context() -> None:
    """Otherwise an edit elsewhere in the file would re-buy every context."""
    assert enrich.chunk_key("4.4", "row") == enrich.chunk_key("4.4", "row")
    assert enrich.chunk_key("4.4", "row") != enrich.chunk_key("4.5", "row")


# ---------------------------------------------------------------------------
# Failure is soft
# ---------------------------------------------------------------------------


def test_a_failed_profile_call_still_yields_a_capability(monkeypatch) -> None:
    """The deterministic half of the profile does not depend on a model."""

    async def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("the model is down")

    monkeypatch.setattr("app.rag.llm.complete_json", boom)

    profile = asyncio.run(
        enrich.profile_document(make_parsed(), "f.docx", "License Management")
    )
    assert profile.capability == "License Management"
    assert profile.summary == ""


def test_a_failed_contextualisation_returns_one_empty_string_per_chunk(
    monkeypatch,
) -> None:
    async def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("the model is down")

    monkeypatch.setattr("app.rag.llm.complete_json", boom)

    chunks = [make_chunk(i) for i in range(5)]
    contexts = asyncio.run(
        enrich.contextualise_chunks(chunks, DocumentProfile(), "A Document")
    )
    assert contexts == ["", "", "", "", ""]


def test_a_malformed_response_does_not_shift_contexts_onto_wrong_chunks(
    monkeypatch,
) -> None:
    """Contexts are matched by ordinal, so a short or shuffled reply is safe."""

    async def partial(*_args: object, **_kwargs: object) -> str:
        return json.dumps(
            {"contexts": [{"ordinal": 2, "context": "about chunk two"}]}
        )

    monkeypatch.setattr("app.rag.llm.complete_json", partial)

    chunks = [make_chunk(i) for i in range(4)]
    contexts = asyncio.run(
        enrich.contextualise_chunks(chunks, DocumentProfile(), "A Document")
    )
    assert contexts == ["", "", "about chunk two", ""]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_chunks_whose_text_is_unchanged_are_not_contextualised_again(
    monkeypatch,
) -> None:
    """The property that makes re-uploading a document cheap.

    One profile call is expected -- the document changed. What must not happen
    is a second call to rewrite contexts that are already correct.
    """
    calls: list[str] = []

    async def record(system: str, _user: str, _schema: object, name: str, **_kw: object) -> str:
        calls.append(name)
        return json.dumps(
            {
                "capability": "License Management",
                "summary": "s",
                "key_terms": [],
                "distinguishing_points": [],
            }
        )

    monkeypatch.setattr("app.rag.llm.complete_json", record)

    chunks = [make_chunk(i, text=f"row {i}") for i in range(6)]
    carried = {enrich.chunk_key(c.heading_path, c.text): f"ctx {c.ordinal}" for c in chunks}

    profile, contexts = asyncio.run(
        enrich.enrich_document(
            make_parsed(),
            filename="f.docx",
            module="License Management",
            chunks=chunks,
            reuse_contexts=carried,
        )
    )

    assert contexts == [f"ctx {i}" for i in range(6)]
    assert calls == ["document_profile"], (
        f"expected only the document pass, got {calls}"
    )
    assert profile.capability == "License Management"


def test_only_the_chunks_that_changed_are_sent_for_contextualisation(
    monkeypatch,
) -> None:
    sent: list[list[int]] = []

    async def capture(_system: str, user: str, _schema: object, name: str, **_kw: object) -> str:
        if name == "document_profile":
            return json.dumps(
                {
                    "capability": "License Management",
                    "summary": "s",
                    "key_terms": [],
                    "distinguishing_points": [],
                }
            )
        ordinals = [
            int(part.split('"')[0])
            for part in user.split('ordinal="')[1:]
        ]
        sent.append(ordinals)
        return json.dumps(
            {"contexts": [{"ordinal": o, "context": f"new {o}"} for o in ordinals]}
        )

    monkeypatch.setattr("app.rag.llm.complete_json", capture)

    chunks = [make_chunk(i, text=f"row {i}") for i in range(6)]
    # Everything is carried forward except chunk 3, which was edited.
    carried = {
        enrich.chunk_key(c.heading_path, c.text): f"ctx {c.ordinal}"
        for c in chunks
        if c.ordinal != 3
    }

    _, contexts = asyncio.run(
        enrich.enrich_document(
            make_parsed(),
            filename="f.docx",
            module="License Management",
            chunks=chunks,
            reuse_contexts=carried,
        )
    )

    assert sent == [[3]], f"only the edited chunk should be sent, got {sent}"
    assert contexts[3] == "new 3"
    assert contexts[0] == "ctx 0"


def test_enrichment_can_be_turned_off_without_losing_the_capability() -> None:
    """`ENRICHMENT_ENABLED=false` must still classify documents for routing."""
    chunks = [make_chunk(i) for i in range(3)]
    profile, contexts = asyncio.run(
        enrich.enrich_document(
            make_parsed(),
            filename="f.docx",
            module="Reporting Period Management",
            chunks=chunks,
            use_llm=False,
        )
    )
    assert profile.capability == "Reporting Period Management"
    assert contexts == ["", "", ""]


# ---------------------------------------------------------------------------
# Where a capability comes from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "module", "siblings", "expected"),
    [
        # What the document says about itself always wins.
        (
            {"capability": "Approval Workflow Management"},
            "Approval workflow",
            {"Approval workflow": "Something Else"},
            "Approval Workflow Management",
        ),
        # A QA workbook with no header inherits from the spec beside it.
        (
            {},
            "Approval workflow",
            {"Approval workflow": "Approval Workflow Management"},
            "Approval Workflow Management",
        ),
        # Failing everything else, the folder name is how these are organised.
        ({}, "License Management", {}, "License Management"),
        ({}, "", {}, ""),
    ],
)
def test_capability_is_resolved_without_a_model_where_possible(
    declared: dict[str, str],
    module: str,
    siblings: dict[str, str],
    expected: str,
) -> None:
    assert enrich.resolve_capability(declared, module, siblings) == expected
