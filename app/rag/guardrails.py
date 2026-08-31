"""Answer-side guardrails: citation validation (G6) and injection detection.

Retrieval decides what the model may see. This module checks what the model
actually did with it.

The important one is `validate_citations`. Every factual sentence must carry a
citation key, and every key must belong to a chunk that was in the authorised
context. A key that is not in that set means one of two things -- the model
invented a source, or it referred to material it should not have -- and both
are answers we refuse to serve. This is cheap, deterministic, and it does not
rely on the model cooperating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.retrieval import RetrievedChunk

# Matches the citation form we ask for: [3fa85f64#7], including multiples
# inside one bracket: [3fa85f64#7, 9c1d2e3f#2]
_CITATION_BLOCK = re.compile(r"\[([0-9a-fA-F]{8}#\d+(?:\s*,\s*[0-9a-fA-F]{8}#\d+)*)\]")

# Phrases that read like an instruction aimed at the model rather than prose
# belonging to a product document. Used for logging and admin review only --
# never to modify or block the answer, because the real defence is that
# document text is fenced and labelled as data (G5).
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (?:all |any )?(?:your |the )?previous instructions",
        r"disregard (?:all |any )?(?:your |the )?(?:previous |prior )?instructions",
        r"you are now (?:a|an|the)\b",
        r"system prompt",
        r"reveal (?:all|the|any) (?:documents|files|secrets|context)",
        r"print (?:all|the entire|every) (?:document|context|file)",
        r"forget (?:everything|all previous)",
        r"act as (?:a|an|the) (?:admin|administrator|root|superuser)",
        r"override (?:your |the )?(?:access|permission|security)",
    )
]


@dataclass(frozen=True, slots=True)
class CitationCheck:
    ok: bool
    cited: set[str]
    valid: set[str]
    invalid: set[str]
    uncited: bool  # answer made claims but cited nothing

    @property
    def has_traceable_support(self) -> bool:
        """At least one claim can be traced to supplied evidence."""
        return bool(self.valid)

    @property
    def failure_reason(self) -> str:
        if self.invalid:
            return f"answer cited keys outside the authorised set: {sorted(self.invalid)}"
        if self.uncited:
            return "answer contained substantive content with no citation"
        return ""


def strip_citations(answer: str, keys: set[str]) -> str:
    """Remove citation markers for the given keys, tidying any empty brackets."""
    if not keys:
        return answer
    lowered = {k.lower() for k in keys}

    def replace(match: re.Match[str]) -> str:
        kept = [
            k.strip()
            for k in match.group(1).split(",")
            if k.strip().lower() not in lowered
        ]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION_BLOCK.sub(replace, answer)
    # Collapse the double spaces a removed marker leaves behind.
    return re.sub(r" {2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",")


def extract_citations(answer: str) -> set[str]:
    keys: set[str] = set()
    for block in _CITATION_BLOCK.findall(answer):
        for key in block.split(","):
            keys.add(key.strip().lower())
    return keys


def validate_citations(
    answer: str, chunks: list[RetrievedChunk], *, refusal_text: str
) -> CitationCheck:
    """Guardrail G6.

    A refusal needs no citations. Anything else must cite, and must cite only
    keys that were actually in the context we supplied.
    """
    allowed = {c.citation_key.lower() for c in chunks}
    cited = extract_citations(answer)
    invalid = cited - allowed
    valid = cited & allowed

    stripped = answer.strip()
    is_refusal = stripped == refusal_text.strip() or not stripped
    # Short acknowledgements are not claims worth citing; anything longer that
    # cites nothing is a synthesis we cannot trace back to a source.
    uncited = not is_refusal and not cited and len(stripped) > 200

    return CitationCheck(
        ok=not invalid and not uncited,
        cited=cited,
        valid=valid,
        invalid=invalid,
        uncited=uncited,
    )


def detect_injection(text: str) -> list[str]:
    """Return the injection-like phrases found in a piece of document text."""
    found: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(match.group(0))
    return found


def scan_chunks_for_injection(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    """Flag retrieved chunks whose text reads like an instruction to the model.

    These are surfaced in the audit console so an administrator can go and
    look at the source file. The answer is still served: the content is
    authorised for this reader, and refusing would let anyone censor the
    assistant by editing a document.
    """
    findings: list[dict[str, object]] = []
    for chunk in chunks:
        hits = detect_injection(chunk.text)
        if hits:
            findings.append(
                {
                    "document_id": str(chunk.document_id),
                    "title": chunk.title,
                    "chunk_ordinal": chunk.ordinal,
                    "phrases": hits,
                }
            )
    return findings
