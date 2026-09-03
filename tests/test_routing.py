"""Capability routing: when to answer, and when to ask.

These are pure functions over retrieval results, so every case here runs
without a database or a model. That is the point of making the ambiguity test
arithmetic rather than another LLM call -- the behaviour is pinned exactly,
and a threshold change shows up as a failing test rather than as a different
answer next Tuesday.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.rag import routing
from app.rag.retrieval import RetrievedChunk

CAPABILITIES = [
    "Approval Workflow Management",
    "Asset Taxonomy & Catalogue",
    "Fields and Screens",
    "License Management",
    "Organization Structure Management",
    "Reporting Period Management",
    "User Access & Permission Management",
]


def chunk(capability: str, score: float, ordinal: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        ordinal=ordinal,
        heading_path="4.1",
        text="body",
        token_count=10,
        title=f"{capability} spec",
        module=capability,
        capability=capability,
        doc_type="Spec",
        sensitivity=2,
        score=score,
    )


def spread(*pairs: tuple[str, float]) -> list[RetrievedChunk]:
    return [chunk(cap, score, i) for i, (cap, score) in enumerate(pairs)]


# ---------------------------------------------------------------------------
# Naming a capability outright
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("In Approval Workflow, what are the open items?", "Approval Workflow Management"),
        ("how do I create a reporting period", "Reporting Period Management"),
        ("asset taxonomy naming rules", "Asset Taxonomy & Catalogue"),
        ("which fields show on the screen", "Fields and Screens"),
        ("user access and permission rules", "User Access & Permission Management"),
    ],
)
def test_a_question_that_names_a_capability_is_scoped_to_it(
    question: str, expected: str
) -> None:
    assert routing.capability_named_in(question, CAPABILITIES) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What are the open items?",
        "What does UAP-FR-045 say?",
        "what are the six access categories?",
        # One shared word is not a name: "asset" must not select Asset Taxonomy.
        "what happens to an asset",
        "",
    ],
)
def test_a_question_that_names_nothing_is_not_scoped(question: str) -> None:
    assert routing.capability_named_in(question, CAPABILITIES) == ""


def test_naming_two_capabilities_names_neither() -> None:
    """Better to ask than to pick whichever sorted first."""
    named = routing.capability_named_in(
        "organization structure and user access permissions", CAPABILITIES
    )
    assert named == ""


def test_a_longer_name_containing_a_shorter_one_is_a_single_mention() -> None:
    caps = ["User Access Management", "User Access & Permission Management"]
    named = routing.capability_named_in(
        "tell me about user access and permission management", caps
    )
    assert named == "User Access & Permission Management"


# ---------------------------------------------------------------------------
# The ambiguity test
# ---------------------------------------------------------------------------


def test_one_capability_holding_the_evidence_is_answered_without_asking() -> None:
    decision = routing.route(
        "what does UAP-FR-045 say?",
        spread(("User Access & Permission Management", 0.9), ("License Management", 0.05)),
    )
    assert not decision.needs_clarification
    assert decision.capability == "User Access & Permission Management"


def test_two_comparable_capabilities_produce_a_question() -> None:
    decision = routing.route(
        "what evidence must be retained?",
        spread(("User Access & Permission Management", 0.34), ("License Management", 0.25)),
    )
    assert decision.needs_clarification
    assert decision.choices == [
        "User Access & Permission Management",
        "License Management",
    ]


def test_evidence_spread_thinly_produces_a_question_even_with_no_close_rival() -> None:
    """The case that slipped through a runner-up-only test.

    44 / 23 / 17 / 16: nothing is within RIVAL_RATIO of the leader, and yet a
    44% leader across four capabilities is exactly the guess to avoid.
    """
    decision = routing.route(
        "what are the open items?",
        spread(
            ("Fields and Screens", 0.44),
            ("Approval Workflow Management", 0.23),
            ("Reporting Period Management", 0.17),
            ("License Management", 0.16),
        ),
    )
    assert decision.needs_clarification
    assert len(decision.choices) == routing.MAX_CANDIDATES


def test_a_clear_leader_with_a_long_tail_is_still_answered() -> None:
    """A dominant capability is not made ambiguous by stray chunks."""
    decision = routing.route(
        "what are the six access categories?",
        spread(
            ("User Access & Permission Management", 0.70),
            ("Organization Structure Management", 0.10),
            ("Fields and Screens", 0.10),
            ("License Management", 0.10),
        ),
    )
    assert not decision.needs_clarification
    assert decision.capability == "User Access & Permission Management"


def test_naming_a_capability_overrides_the_ambiguity_test() -> None:
    """Someone who said which area they meant is not asked again."""
    decision = routing.route(
        "In Approval Workflow, what are the open items?",
        spread(
            ("Fields and Screens", 0.44),
            ("Approval Workflow Management", 0.23),
            ("Reporting Period Management", 0.17),
            ("License Management", 0.16),
        ),
    )
    assert not decision.needs_clarification
    assert decision.capability == "Approval Workflow Management"
    assert decision.named


def test_a_single_matching_capability_is_never_ambiguous() -> None:
    decision = routing.route(
        "what are the test cases?", spread(("Fields and Screens", 0.5))
    )
    assert not decision.needs_clarification
    assert decision.capability == "Fields and Screens"


def test_no_retrieval_routes_nowhere() -> None:
    decision = routing.route("anything", [])
    assert not decision.needs_clarification
    assert decision.capability == ""


def test_never_more_than_three_choices_are_offered() -> None:
    decision = routing.route(
        "what are the open items?",
        spread(
            ("Fields and Screens", 0.21),
            ("Approval Workflow Management", 0.20),
            ("Reporting Period Management", 0.20),
            ("License Management", 0.20),
            ("Asset Taxonomy & Catalogue", 0.19),
        ),
    )
    assert decision.needs_clarification
    assert len(decision.choices) <= routing.MAX_CANDIDATES


# ---------------------------------------------------------------------------
# What the user sees
# ---------------------------------------------------------------------------


def test_the_clarifying_question_names_every_choice() -> None:
    decision = routing.route(
        "what evidence must be retained?",
        spread(("User Access & Permission Management", 0.34), ("License Management", 0.25)),
    )
    text = routing.clarification_question(decision.candidates)
    assert "User Access & Permission Management" in text
    assert "License Management" in text
    assert text.endswith("?")


def test_the_clarifying_question_does_not_guess() -> None:
    """It must ask, not lead. A stated preference here is the old behaviour."""
    decision = routing.route(
        "what are the open items?",
        spread(
            ("Fields and Screens", 0.44),
            ("Approval Workflow Management", 0.23),
            ("Reporting Period Management", 0.17),
        ),
    )
    text = routing.clarification_question(decision.candidates).lower()
    for lead in ("probably", "likely", "i think", "most likely", "i'll assume"):
        assert lead not in text


# ---------------------------------------------------------------------------
# The audit trail
# ---------------------------------------------------------------------------


def test_every_decision_carries_a_reason() -> None:
    """"Why did it ask me that?" has to be answerable from the log."""
    cases = [
        spread(("Fields and Screens", 0.5)),
        spread(("Fields and Screens", 0.9), ("License Management", 0.05)),
        spread(("Fields and Screens", 0.34), ("License Management", 0.30)),
        [],
    ]
    for chunks in cases:
        assert routing.route("a question", chunks).reason


# ---------------------------------------------------------------------------
# Choosing an area, and choosing all of them
# ---------------------------------------------------------------------------


def test_the_all_areas_sentinel_is_not_passed_to_the_search() -> None:
    """As a filter value it would match nothing and refuse every question."""
    from app.rag.answer import search_scope

    assert search_scope(routing.ALL_CAPABILITIES) == ""
    assert search_scope("") == ""
    assert search_scope("License Management") == "License Management"


def test_choosing_all_areas_does_not_ask_again() -> None:
    """Otherwise the button loops: the same question clarifies every time."""
    from app.rag.answer import route_question
    from app.rag.retrieval import RetrievalResult

    context = RetrievalResult(
        chunks=spread(
            ("Fields and Screens", 0.44),
            ("Approval Workflow Management", 0.23),
            ("Reporting Period Management", 0.17),
        ),
        blocked=[],
        anomalies=[],
    )

    _, decision = asyncio.run(
        route_question(
            None,  # type: ignore[arg-type]  # not reached: the scope short-circuits
            None,  # type: ignore[arg-type]
            "what are the open items?",
            context,
            routing.ALL_CAPABILITIES,
        )
    )
    assert not decision.needs_clarification
    assert decision.capability == ""


def test_a_chosen_area_is_taken_at_face_value() -> None:
    """The retrieval was already scoped to it; do not re-route."""
    from app.rag.answer import route_question
    from app.rag.retrieval import RetrievalResult

    context = RetrievalResult(
        chunks=spread(("Approval Workflow Management", 0.5)),
        blocked=[],
        anomalies=[],
    )

    _, decision = asyncio.run(
        route_question(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            "what are the open items?",
            context,
            "Approval Workflow Management",
        )
    )
    assert not decision.needs_clarification
    assert decision.capability == "Approval Workflow Management"
