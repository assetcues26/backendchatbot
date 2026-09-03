"""Prompts used at ingest time, kept apart from the answering prompts.

Everything here writes text that is embedded but never displayed. That is what
makes it safe to let a model write into the retrieval path at all, and it is
also why these prompts are so insistent about using nothing but the supplied
document: a sentence invented here can never become a quoted claim, but it can
still drag a chunk to the wrong place in the index.
"""

from __future__ import annotations

DOCUMENT_PROFILE_SYSTEM_PROMPT = """\
You describe one internal product document so a search system can tell it \
apart from its siblings.

AssetCues documents are written to a shared template, so several of them look \
alike at a glance. Your job is to capture what makes THIS one distinct.

Rules:
- Use ONLY the supplied document. Do not add anything you know about asset \
management, about AssetCues, or about software generally. If the document does \
not say it, it does not go in.
- summary: what the document governs and who acts on it, in three or four \
sentences.
- key_terms: the product terms this document defines or governs, for example \
"Access Category", "Permission Group", "Reporting Period". Terms it merely \
mentions in passing do not count.
- distinguishing_points: the things someone could otherwise confuse with a \
sibling document -- the specific scope, the specific rules, the boundary it \
draws. "It is a specification" is useless. "It defines the six Access \
Categories and the administrator hierarchy, but not approval routing" is \
useful.
- capability: the product capability this document belongs to, exactly as the \
document names it. If the document does not name one, infer the narrowest \
capability its content covers.\
"""

DOCUMENT_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "capability": {
            "type": "string",
            "description": "Product capability this document belongs to.",
        },
        "summary": {"type": "string"},
        "key_terms": {"type": "array", "items": {"type": "string"}},
        "distinguishing_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["capability", "summary", "key_terms", "distinguishing_points"],
    "additionalProperties": False,
}


CHUNK_CONTEXT_SYSTEM_PROMPT = """\
You write one short passage per excerpt, saying where it sits in its document \
and what it covers.

This text is used only to help a search system find the right excerpt. It is \
never shown to anyone and never quoted, so write for retrieval rather than for \
reading.

Rules:
- Use ONLY the document details supplied. Never add outside knowledge, and \
never state a fact the excerpt does not support.
- Name the capability and the section. Excerpts from different capabilities are \
often word-for-word identical -- a table header like "Requirement | Mapped \
tests" appears in many documents -- and your passage is the only thing that \
makes this one findable. That is the whole problem you are solving.
- Say what the excerpt is about in concrete terms: which requirements, which \
rules, which identifiers appear in it.
- One or two sentences. No preamble, and do not begin with "This excerpt".
- Return one entry per excerpt, keeping its ordinal.\
"""

CHUNK_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "contexts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ordinal": {"type": "integer"},
                    "context": {"type": "string"},
                },
                "required": ["ordinal", "context"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["contexts"],
    "additionalProperties": False,
}


def build_document_profile_prompt(
    *,
    title: str,
    filename: str,
    module: str,
    doc_type: str,
    declared: dict[str, str],
    excerpt: str,
) -> str:
    stated = "\n".join(f"  {k}: {v}" for k, v in declared.items()) or "  (none)"
    return (
        f"TITLE: {title}\n"
        f"FILENAME: {filename}\n"
        f"SOURCE FOLDER: {module or '(none)'}\n"
        f"DOCUMENT TYPE: {doc_type}\n"
        f"DECLARED IN THE DOCUMENT HEADER:\n{stated}\n\n"
        f"DOCUMENT:\n{excerpt}"
    )


def build_chunk_context_prompt(
    *,
    title: str,
    capability: str,
    summary: str,
    key_terms: list[str],
    chunks: list[tuple[int, str, str]],
) -> str:
    terms = ", ".join(key_terms[:20]) or "(none recorded)"
    blocks = [
        f'<excerpt ordinal="{ordinal}" section="{_escape(heading)}">\n'
        f"{text[:2200]}\n</excerpt>"
        for ordinal, heading, text in chunks
    ]
    return (
        f"DOCUMENT: {title}\n"
        f"CAPABILITY: {capability or '(not stated)'}\n"
        f"SUMMARY: {summary or '(none)'}\n"
        f"TERMS THIS DOCUMENT DEFINES: {terms}\n\n"
        "EXCERPTS\n" + "\n\n".join(blocks) + "\n\n"
        "Write a situating passage for each excerpt, by ordinal."
    )


def _escape(value: str) -> str:
    """Keep a heading from breaking out of its own attribute."""
    return value.replace('"', "'").replace("<", "(").replace(">", ")")
