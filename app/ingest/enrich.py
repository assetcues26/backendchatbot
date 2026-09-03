"""Contextual enrichment: teach each chunk where it lives.

The problem this exists to solve is measurable. Every AssetCues document is
written to one template, so its scaffolding is near-identical across
capabilities. Sampling the live index for each chunk's nearest neighbour in a
*different* module found pairs at 0.98 cosine similarity -- rows like
``| Requirement | Mapped tests | Authoritative definition |`` that are
byte-identical in two modules. No retriever can separate those, because as
text they are not different. What distinguishes them is the document they sit
in, and the chunk carried none of that.

So each chunk gets a short passage saying where it comes from and what it is
about, and that passage is embedded *with* the text.

Two rules hold this together, and both matter:

1. **The context is embedded, never shown and never citable.** It steers
   ranking; it cannot reach a reader as though the document said it. Answers
   and citations still quote the original text only. That is what makes it
   safe to let a model write into the retrieval path at all.

2. **The context may only restate what is already in the corpus.** The prompt
   forbids outside knowledge. Combined with rule 1, the worst a hallucinated
   sentence can do is rank a chunk oddly -- it can never become a claim.

Enrichment is best-effort throughout. Every failure path leaves the chunk
indexed from its own text: worse ranking, never a lost document.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field

from app.ingest.chunker import Chunk as TextChunk
from app.ingest.parsers import ParsedDocument, extract_header_fields
from app.rag import ingest_prompts as prompts
from app.rag import llm

# Chunks per contextualisation call. One call per chunk would mean 779 calls
# for this corpus; batching brings it under a hundred while keeping each
# request small enough that the model attends to every chunk in it.
BATCH_SIZE = 8

# How much of the document the understanding pass reads. The AssetCues
# template front-loads purpose, audience and reading rules, and the headings
# carry the rest of the shape.
_SUMMARY_CHARS = 14000

# Concurrency for batched calls. Enough to keep a backfill brisk, low enough
# not to trip rate limits on a single key.
_MAX_PARALLEL = 4


@dataclass(slots=True)
class DocumentProfile:
    """What a document is, as far as the corpus itself can say."""

    capability: str = ""
    module_declared: str = ""
    product_domain: str = ""
    summary: str = ""
    key_terms: list[str] = field(default_factory=list)
    distinguishing_points: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.summary or self.capability)


def resolve_capability(
    declared: dict[str, str], module: str, siblings: dict[str, str]
) -> str:
    """Decide a document's capability without asking a model.

    In order of trust:

    1. What the document declares in its own header table.
    2. What a sibling in the same folder declares -- the QA workbook beside a
       specification is the same capability whether or not it says so.
    3. The folder name, which is how these files are already organised.

    Only if all three come up empty is it worth spending a model call.
    """
    if declared.get("capability"):
        return declared["capability"]
    if module and siblings.get(module):
        return siblings[module]
    return module


async def profile_document(
    parsed: ParsedDocument,
    filename: str,
    module: str,
    sibling_capabilities: dict[str, str] | None = None,
) -> DocumentProfile:
    """Read a document once and describe it. Never raises."""
    declared = extract_header_fields(parsed.markdown)
    capability = resolve_capability(declared, module, sibling_capabilities or {})

    profile = DocumentProfile(
        capability=capability,
        module_declared=declared.get("module_declared", ""),
        product_domain=declared.get("product_domain", ""),
    )

    try:
        raw = await llm.complete_json(
            prompts.DOCUMENT_PROFILE_SYSTEM_PROMPT,
            prompts.build_document_profile_prompt(
                title=parsed.title,
                filename=filename,
                module=module,
                doc_type=parsed.doc_type,
                declared=declared,
                excerpt=parsed.markdown[:_SUMMARY_CHARS],
            ),
            prompts.DOCUMENT_PROFILE_SCHEMA,
            "document_profile",
            max_tokens=900,
        )
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - a profile is an improvement, not a step
        return profile

    # The declared capability always wins over an inferred one: the document
    # saying what it is beats a model guessing.
    if not profile.capability:
        profile.capability = str(data.get("capability") or "").strip()[:200]

    profile.summary = str(data.get("summary") or "").strip()[:2000]
    profile.key_terms = _string_list(data.get("key_terms"), limit=25, max_len=120)
    profile.distinguishing_points = _string_list(
        data.get("distinguishing_points"), limit=8, max_len=300
    )
    return profile


async def contextualise_chunks(
    chunks: list[TextChunk], profile: DocumentProfile, title: str
) -> list[str]:
    """A situating passage per chunk, in order. Never raises.

    Returns a list the same length as `chunks`; entries are empty strings
    wherever the model could not help, which the caller treats as "embed the
    text alone".
    """
    if not chunks:
        return []

    batches = [
        chunks[start : start + BATCH_SIZE]
        for start in range(0, len(chunks), BATCH_SIZE)
    ]
    semaphore = asyncio.Semaphore(_MAX_PARALLEL)

    async def run(batch: list[TextChunk]) -> list[str]:
        async with semaphore:
            return await _contextualise_batch(batch, profile, title)

    results = await asyncio.gather(
        *(run(batch) for batch in batches), return_exceptions=True
    )

    out: list[str] = []
    for batch, result in zip(batches, results, strict=True):
        if isinstance(result, BaseException) or not isinstance(result, list):
            out.extend("" for _ in batch)
        else:
            out.extend(result[: len(batch)])
            out.extend("" for _ in range(len(batch) - len(result)))
    return out[: len(chunks)]


async def _contextualise_batch(
    batch: list[TextChunk], profile: DocumentProfile, title: str
) -> list[str]:
    try:
        raw = await llm.complete_json(
            prompts.CHUNK_CONTEXT_SYSTEM_PROMPT,
            prompts.build_chunk_context_prompt(
                title=title,
                capability=profile.capability,
                summary=profile.summary,
                key_terms=profile.key_terms,
                chunks=[(c.ordinal, c.heading_path, c.text) for c in batch],
            ),
            prompts.CHUNK_CONTEXT_SCHEMA,
            "chunk_contexts",
            max_tokens=180 * len(batch) + 200,
        )
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - fall back to the text alone
        return ["" for _ in batch]

    by_ordinal = {}
    for item in data.get("contexts", []):
        if not isinstance(item, dict):
            continue
        try:
            ordinal = int(item["ordinal"])
        except (KeyError, TypeError, ValueError):
            continue
        by_ordinal[ordinal] = str(item.get("context") or "").strip()[:600]

    return [by_ordinal.get(c.ordinal, "") for c in batch]


def embedding_text(context: str, heading_path: str, text: str) -> str:
    """What actually gets embedded.

    Context first: it is the part that distinguishes this chunk from the
    identical-looking row in another capability, and putting it first means it
    survives any truncation. The reader never sees this string.
    """
    parts = [p for p in (context.strip(), heading_path.strip(), text.strip()) if p]
    return "\n\n".join(parts)


def _string_list(value: object, *, limit: int, max_len: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text[:max_len])
    return out[:limit]


def chunk_key(heading_path: str, text: str) -> str:
    """Identity of a chunk's *source text*, ignoring any context written for it.

    Used to carry a context forward across a re-upload: if the author fixed a
    typo on page 3, the other 60 chunks keep the context they already have and
    cost nothing. Distinct from the stored ``text_sha256``, which covers the
    context too so that a *changed* context correctly invalidates the cached
    embedding.
    """
    return hashlib.sha256(f"{heading_path}\n{text}".encode()).hexdigest()


def chunk_digest(context: str, heading_path: str, text: str) -> str:
    """Identity of a chunk *as embedded*.

    Covers the context as well as the text, so rewriting a context
    correctly invalidates the cached embedding. Compare `chunk_key`, which
    covers the source text alone and is what carries a context forward
    across an edit.
    """
    return hashlib.sha256("\n".join((context, heading_path, text)).encode()).hexdigest()


async def enrich_document(
    parsed: ParsedDocument,
    *,
    filename: str,
    module: str,
    chunks: list[TextChunk],
    siblings: dict[str, str] | None = None,
    reuse_contexts: dict[str, str] | None = None,
    use_llm: bool = True,
) -> tuple[DocumentProfile, list[str]]:
    """Profile a document and situate each of its chunks.

    Returns the profile and one context string per chunk, in order. With
    `use_llm=False` the profile is still filled in from the document's own
    header table -- that part costs nothing and is more trustworthy than any
    model output -- and every context comes back empty.

    Never raises: ingest must not fail because an enrichment call did.
    """
    declared = extract_header_fields(parsed.markdown)

    if use_llm:
        profile = await profile_document(parsed, filename, module, siblings)
    else:
        profile = DocumentProfile(
            capability=resolve_capability(declared, module, siblings or {}),
            module_declared=declared.get("module_declared", ""),
            product_domain=declared.get("product_domain", ""),
        )

    carried = reuse_contexts or {}
    contexts = [carried.get(chunk_key(c.heading_path, c.text), "") for c in chunks]

    missing = [c for c, existing in zip(chunks, contexts, strict=True) if not existing]
    if use_llm and missing:
        written = await contextualise_chunks(missing, profile, parsed.title)
        fresh = {
            c.ordinal: text
            for c, text in zip(missing, written, strict=True)
        }
        contexts = [
            existing or fresh.get(c.ordinal, "")
            for c, existing in zip(chunks, contexts, strict=True)
        ]

    return profile, contexts
