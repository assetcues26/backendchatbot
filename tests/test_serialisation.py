"""Serialisation of the dataclasses that cross the API boundary.

These exist because a bug here took down the entire chat page while every
other test passed. `Citation` and `AnswerResult` are declared `slots=True`,
so they have no `__dict__`, and `c.__dict__` raises AttributeError. The unit
tests called `answer_question` directly and never serialised its result, so
nothing noticed until a real browser hit the endpoint.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, fields, replace

from app.api.schemas import CitationOut
from app.rag.answer import AnswerResult, Citation


def make_citation() -> Citation:
    return Citation(
        key="3fa85f64#7",
        document_id=str(uuid.uuid4()),
        title="User Access & Permission Management",
        doc_type="Product & Functional Specification",
        module="User Access Management",
        heading_path="4.3 Permission Groups",
        ordinal=7,
    )


def test_citation_has_no_dict_so_asdict_is_required() -> None:
    """Guards the exact mistake: reaching for __dict__ on a slots dataclass."""
    citation = make_citation()
    assert not hasattr(citation, "__dict__")
    assert asdict(citation)["key"] == "3fa85f64#7"


def test_answer_result_has_no_dict_either() -> None:
    assert not hasattr(AnswerResult(answer="x"), "__dict__")


def test_a_citation_converts_into_the_api_schema() -> None:
    """This is what /api/ask and /api/compare do on every request."""
    out = CitationOut(**asdict(make_citation()))
    assert out.key == "3fa85f64#7"
    assert out.ordinal == 7


def test_citation_fields_match_the_api_schema_exactly() -> None:
    """`CitationOut(**asdict(c))` fails loudly if either side gains a field."""
    assert {f.name for f in fields(Citation)} == set(CitationOut.model_fields)


def test_a_cached_result_is_copied_not_mutated() -> None:
    """The cache hands out shared objects; marking one cached must not edit
    the stored entry, and `replace` is what makes that safe on a slots class."""
    original = AnswerResult(answer="hello", citations=[make_citation()])
    cached = replace(original, cached=True)

    assert cached.cached is True
    assert original.cached is False
    assert cached.answer == "hello"
    # Citations must survive as objects, not be flattened into dicts.
    assert isinstance(cached.citations[0], Citation)


def test_the_stream_done_payload_is_json_serialisable() -> None:
    """The `done` SSE event serialises citations; asdict keeps that valid."""
    import json

    payload = {"citations": [asdict(make_citation())], "refused": False}
    assert json.loads(json.dumps(payload))["citations"][0]["ordinal"] == 7
