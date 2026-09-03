"""Deciding which part of the product a question is about.

Eight capabilities are documented, several of them written to the same
template, and a question like "what are the open items?" is a fair question
about any of them. Answering it from whichever one happened to rank first is
the failure this module exists to prevent: a confident answer, correctly
cited, about the wrong capability.

So there are three outcomes, in this order:

1. **The question names a capability.** "In Approval Workflow, what are the
   open items?" -- scope to it and answer.
2. **One capability clearly wins the retrieval.** Its share of the fused score
   is large enough that the alternatives are noise -- answer, unscoped, and
   note which capability it was.
3. **Two or more are comparable.** Ask which one they meant.

The third case is the whole point, and it is decided **arithmetically** -- a
share-of-score threshold over results the retriever already produced. No extra
model call, no latency, and the number that triggered it is written to the
audit log, so "why did it ask me that?" has an answer.

Nothing here is a security control. Scoping only ever narrows a search the
access predicate has already restricted (see `SCOPED_DOCS_CTE`), and the
candidates offered back to a user are derived from chunks they were allowed to
retrieve -- so a clarifying question can never name a capability they cannot
read.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.rag.retrieval import RetrievedChunk

# One capability must hold this share of the fused score to answer without
# asking. Two-thirds: comfortably more than an even split between two, and
# reachable by a genuine winner when three or four capabilities match weakly.
DOMINANT_SHARE = 0.62

# A runner-up this close to the leader means the retrieval genuinely could not
# separate them. Expressed as a ratio of the leader's score rather than an
# absolute gap, because RRF scores are small and their scale varies with how
# many results came back.
RIVAL_RATIO = 0.55

# Below this share a capability is a stray chunk, not a candidate worth
# putting in front of someone.
MIN_CANDIDATE_SHARE = 0.15

# A leader below DOMINANT_SHARE with this many capabilities still in play
# is a spread, not a winner -- even when no single rival is close to it.
# "What are the open items?" scored 44 / 23 / 17 / 16 across four
# capabilities: nothing was within RIVAL_RATIO of the leader, and yet
# answering from a 44% leader is exactly the guess being removed.
SPREAD_CANDIDATES = 3

# How much of a capability's distinctive wording a question must contain
# before it counts as naming it.
MIN_NAME_MATCH = 0.6

# What a caller sends to mean "I saw the choices and I want all of them".
#
# It has to be distinct from "no capability given", because those two are not
# the same request: an empty scope is a question that has not been routed yet
# and may still be answered with a clarification, whereas this one has already
# been asked and answered. Without it, clicking "answer from all of them"
# would re-ask a question that clarifies again, forever.
#
# It is never used as a filter value -- the caller translates it to no scope
# before retrieval -- so it cannot match a document.
ALL_CAPABILITIES = "*"

# Never ask someone to choose between more than this many. Past three it stops
# being a question and starts being a menu.
MAX_CANDIDATES = 3


@dataclass(frozen=True, slots=True)
class CapabilityScore:
    capability: str
    score: float
    share: float
    chunk_count: int
    titles: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Route:
    """What to do with a question, and why."""

    capability: str = ""
    needs_clarification: bool = False
    candidates: list[CapabilityScore] = field(default_factory=list)
    named: bool = False
    reason: str = ""

    @property
    def choices(self) -> list[str]:
        return [c.capability for c in self.candidates]


def normalise(value: str) -> str:
    """Fold a capability name to something a question can be matched against."""
    text = unicodedata.normalize("NFKD", value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _content_words(value: str) -> list[str]:
    # "Management", "Configuration" and friends appear in most capability
    # names, so matching on them identifies nothing.
    stop = {
        "and",
        "the",
        "of",
        "management",
        "managment",
        "configuration",
        "control",
        "controls",
        "system",
        "module",
    }
    return [w for w in normalise(value).split() if w not in stop and len(w) > 2]


def _mentions(word: str, asked: list[str]) -> bool:
    """Is this word of a capability name present in the question?

    Prefix matching in both directions, which is enough stemming for the words
    that actually occur here: "screen" finds "screens", "report" finds
    "reporting". Below four characters it has to be exact, or "use" would find
    "user".
    """
    for spoken in asked:
        if word == spoken:
            return True
        shorter, longer = sorted((word, spoken), key=len)
        if len(shorter) >= 4 and longer.startswith(shorter):
            return True
    return False


def capability_named_in(question: str, capabilities: list[str]) -> str:
    """The capability the question names outright, if it names one.

    Matches the full name first, then falls back to the distinctive words in
    it, so "approval workflow" finds "Approval Workflow Management" and
    "licensing" does not.

    Ambiguous mentions return nothing: naming two capabilities is not naming
    one, and the caller should ask rather than pick.
    """
    asked = normalise(question)
    if not asked:
        return ""

    exact = [c for c in capabilities if c and normalise(c) in asked]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # Prefer the longest match: "User Access Management" inside "User
        # Access & Permission Management" is the same mention, not two.
        longest = max(exact, key=lambda c: len(normalise(c)))
        if all(normalise(c) in normalise(longest) for c in exact):
            return longest
        return ""

    asked_words = asked.split()
    matched = []
    for capability in capabilities:
        words = _content_words(capability)
        if not words:
            continue
        hits = [w for w in words if _mentions(w, asked_words)]
        # Most of the name has to be there, and one word is never enough for a
        # multi-word name: "asset" alone must not select Asset Taxonomy over
        # Asset Register.
        enough = len(hits) >= max(2, round(MIN_NAME_MATCH * len(words))) or (
            len(words) == 1 and len(hits) == 1
        )
        if enough:
            matched.append(capability)

    # Naming two capabilities is not naming one. Fall through and let the
    # caller ask, rather than picking whichever sorted first.
    return matched[0] if len(matched) == 1 else ""


def score_capabilities(chunks: list[RetrievedChunk]) -> list[CapabilityScore]:
    """Fused retrieval score per capability, strongest first."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    titles: dict[str, list[str]] = {}

    for chunk in chunks:
        key = chunk.capability or chunk.module
        if not key:
            continue
        totals[key] = totals.get(key, 0.0) + chunk.score
        counts[key] = counts.get(key, 0) + 1
        if chunk.title not in titles.setdefault(key, []):
            titles[key].append(chunk.title)

    grand = sum(totals.values())
    if grand <= 0:
        return []

    scored = [
        CapabilityScore(
            capability=key,
            score=total,
            share=total / grand,
            chunk_count=counts[key],
            titles=titles[key][:3],
        )
        for key, total in totals.items()
    ]
    scored.sort(key=lambda c: (-c.score, c.capability))
    return scored


def route(question: str, chunks: list[RetrievedChunk]) -> Route:
    """Decide whether to answer, and from where.

    `chunks` are already access-filtered, so every capability considered here
    is one the caller may read.
    """
    scored = score_capabilities(chunks)
    if not scored:
        return Route(reason="no retrieval to route on")

    named = capability_named_in(question, [c.capability for c in scored])
    if named:
        return Route(
            capability=named,
            named=True,
            candidates=scored,
            reason="named in the question",
        )

    leader = scored[0]
    if len(scored) == 1:
        return Route(
            capability=leader.capability,
            candidates=scored,
            reason="only one capability matched",
        )

    if leader.share >= DOMINANT_SHARE:
        return Route(
            capability=leader.capability,
            candidates=scored,
            reason=f"clear winner at {leader.share:.0%} of the score",
        )

    # Two ways a question can turn out to straddle capabilities.
    rivals = [
        c
        for c in scored[1:]
        if c.score >= leader.score * RIVAL_RATIO and c.share >= MIN_CANDIDATE_SHARE
    ]
    in_play = [c for c in scored if c.share >= MIN_CANDIDATE_SHARE]

    if rivals:
        return Route(
            needs_clarification=True,
            candidates=[leader, *rivals][:MAX_CANDIDATES],
            reason=(
                f"{len(rivals) + 1} capabilities within {RIVAL_RATIO:.0%} of "
                f"each other; leader holds only {leader.share:.0%}"
            ),
        )

    if len(in_play) >= SPREAD_CANDIDATES:
        return Route(
            needs_clarification=True,
            candidates=in_play[:MAX_CANDIDATES],
            reason=(
                f"the evidence is spread over {len(in_play)} capabilities; "
                f"leader holds only {leader.share:.0%}"
            ),
        )

    return Route(
        capability=leader.capability,
        candidates=scored,
        reason=f"leader at {leader.share:.0%}, no comparable rival",
    )


def clarification_question(candidates: list[CapabilityScore]) -> str:
    """The sentence a user reads when a question straddles capabilities.

    Deliberately plain, and deliberately does not guess. It says the question
    fits more than one area and asks which, because inventing a preference
    here is exactly the behaviour being removed.
    """
    names = [c.capability for c in candidates]
    if len(names) == 2:
        listed = f"{names[0]} or {names[1]}"
    else:
        listed = ", ".join(names[:-1]) + f", or {names[-1]}"
    return (
        f"That question fits more than one part of the product. "
        f"Did you mean {listed}?"
    )
