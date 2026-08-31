"""Conversation memory and follow-up suggestions.

The security-relevant part is what conversation memory is allowed to be: the
caller's own earlier questions, never assistant answers. A client can put
anything in a request body, and fabricated "the assistant said X" text would
land straight in the prompt.
"""

from __future__ import annotations

import uuid

import pytest

from app.api.schemas import AskRequest
from app.rag.answer import MAX_HISTORY, condense_question, looks_like_a_follow_up
from app.rag.retrieval import RetrievedChunk


def make_chunk(ordinal: int, title: str = "User Access") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        ordinal=ordinal,
        heading_path="4.3 Permission Groups",
        text="body",
        token_count=10,
        title=title,
        module="m",
        doc_type="Product & Functional Specification",
        sensitivity=3,
        score=0.5,
    )


# ---------------------------------------------------------------------------
# What history is allowed to contain
# ---------------------------------------------------------------------------


def test_history_accepts_only_a_list_of_questions() -> None:
    """No slot exists for an assistant turn, so none can be smuggled in."""
    assert set(AskRequest.model_fields) == {"question", "history"}
    request = AskRequest(question="what about sales?", history=["Who is admin?"])
    assert request.history == ["Who is admin?"]


def test_history_is_capped_so_the_prompt_cannot_grow_without_bound() -> None:
    with pytest.raises(ValueError):
        AskRequest(question="q", history=[f"q{i}" for i in range(MAX_HISTORY + 1)])


def test_history_carries_no_identity_claim() -> None:
    """Guardrail G2 still holds for the conversational path."""
    forbidden = {"role", "roles", "tenant_id", "clearance", "user_id"}
    assert not (set(AskRequest.model_fields) & forbidden)


# ---------------------------------------------------------------------------
# The rewrite gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "what about for sales?",
        "And QA?",
        "why is that?",
        "How about those permission groups?",
    ],
)
def test_short_or_referential_questions_are_treated_as_follow_ups(
    question: str,
) -> None:
    assert looks_like_a_follow_up(question)


def test_a_long_standalone_question_skips_the_rewrite() -> None:
    """The gate exists to avoid paying for a model call on every turn.

    'Entitlement' contains the letters 'it', which is why the word-boundary
    anchors in the pattern are load-bearing.
    """
    assert not looks_like_a_follow_up(
        "Explain in full detail how the Partner Entitlement Envelope constrains "
        "partner allocations across every managed customer"
    )


async def test_no_history_means_no_rewrite_and_no_model_call() -> None:
    question = "What are the six Access Categories?"
    assert await condense_question(question, []) == question


async def test_blank_history_entries_are_ignored() -> None:
    question = "What are the six Access Categories?"
    assert await condense_question(question, ["  ", ""]) == question


# ---------------------------------------------------------------------------
# Answer feedback
# ---------------------------------------------------------------------------


def test_feedback_references_a_turn_rather_than_restating_it() -> None:
    """The question and sources must come from the server's own audit row.

    If the client supplied them, feedback could be filed against a question
    nobody asked. `turn_id` is the query's audit row; the route reads the rest
    back from it.
    """
    from app.api.schemas import FeedbackIn

    assert "turn_id" in FeedbackIn.model_fields
    assert "question" not in FeedbackIn.model_fields
    assert "documents_used" not in FeedbackIn.model_fields


def test_feedback_carries_no_identity_claim() -> None:
    """G2 again: who filed it comes from the JWT, not the body."""
    from app.api.schemas import FeedbackIn

    forbidden = {"role", "roles", "tenant_id", "clearance", "user_id", "email"}
    assert not (set(FeedbackIn.model_fields) & forbidden)


@pytest.mark.parametrize("rating", ["up", "down"])
def test_valid_ratings_are_accepted(rating: str) -> None:
    from app.api.schemas import FeedbackIn

    assert FeedbackIn(turn_id=uuid.uuid4(), rating=rating).rating == rating


def test_an_arbitrary_rating_is_rejected() -> None:
    from app.api.schemas import FeedbackIn

    with pytest.raises(ValueError):
        FeedbackIn(turn_id=uuid.uuid4(), rating="sideways")


def test_a_turn_id_must_be_a_real_identifier() -> None:
    from app.api.schemas import FeedbackIn

    with pytest.raises(ValueError):
        FeedbackIn(turn_id="not-a-uuid", rating="up")


def test_the_answer_field_is_bounded() -> None:
    """Client-reported text still goes in the audit log; cap what it can be."""
    from app.api.schemas import FeedbackIn

    with pytest.raises(ValueError):
        FeedbackIn(turn_id=uuid.uuid4(), rating="down", answer="x" * 8001)
