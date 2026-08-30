"""Structure-aware chunking.

Three rules, each of which exists because breaking it produced visibly worse
answers on this corpus:

1. **Split on headings first.** A chunk that spans "4.3 Permission Groups" and
   "4.4 Profiles" retrieves for both and answers neither well.

2. **Never split a markdown table.** The AssetCues specifications put their
   substance in tables -- requirement id, rule, acceptance criterion, one per
   row. Cutting a table mid-row separates "UAP-FR-045" from the rule it
   identifies, which is precisely the association a reader is asking about.
   When a table is larger than the target size it is split on row boundaries
   and the header row is repeated, so every piece stays self-describing.

3. **Carry the heading path.** Citations read "4.3 Permission Groups" instead
   of "chunk 47", and the path is prepended to the embedded text so the
   section name itself is searchable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

TARGET_TOKENS = 450
OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 24  # below this a chunk is noise; merge it forward

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    heading_path: str
    text: str
    token_count: int

    @property
    def sha256(self) -> str:
        # Hash the heading path too: moving a paragraph to a different section
        # changes its meaning and must invalidate the cached embedding.
        return hashlib.sha256(
            f"{self.heading_path}\n{self.text}".encode()
        ).hexdigest()


@lru_cache(maxsize=1)
def _encoder() -> object | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # pragma: no cover - tiktoken is optional at runtime
        return None


def count_tokens(text: str) -> int:
    encoder = _encoder()
    if encoder is None:
        # ~0.75 words per token is a good enough fallback for chunk sizing.
        return max(1, int(len(text.split()) / 0.75))
    return len(encoder.encode(text))  # type: ignore[attr-defined]


@dataclass(slots=True)
class _Block:
    """A heading, a paragraph, or a whole table -- an atom we prefer to keep."""

    text: str
    heading_path: str
    is_table: bool
    tokens: int


def _split_into_blocks(markdown: str) -> list[_Block]:
    lines = markdown.splitlines()
    blocks: list[_Block] = []
    stack: list[str] = []  # current heading path, one entry per level

    buffer: list[str] = []
    table_buffer: list[str] = []

    def path() -> str:
        return " > ".join(stack)

    def flush_text() -> None:
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            blocks.append(_Block(text, path(), False, count_tokens(text)))

    def flush_table() -> None:
        if not table_buffer:
            return
        text = "\n".join(table_buffer).strip()
        table_buffer.clear()
        if text:
            blocks.append(_Block(text, path(), True, count_tokens(text)))

    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            flush_text()
            flush_table()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            del stack[level - 1 :]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            continue

        if _TABLE_ROW.match(line):
            flush_text()
            table_buffer.append(line)
            continue

        flush_table()
        if line.strip():
            buffer.append(line)
        else:
            flush_text()

    flush_text()
    flush_table()
    return blocks


def _table_header(table_text: str) -> list[str]:
    """Header row plus separator, to repeat when a big table is split."""
    rows = table_text.splitlines()
    if len(rows) >= 2 and _TABLE_SEPARATOR.match(rows[1]):
        return rows[:2]
    return rows[:1]


def _split_table(block: _Block) -> list[_Block]:
    """Break an oversized table on row boundaries, repeating the header."""
    rows = block.text.splitlines()
    header = _table_header(block.text)
    body = rows[len(header) :]
    header_tokens = count_tokens("\n".join(header))

    pieces: list[_Block] = []
    current: list[str] = []
    current_tokens = header_tokens

    for row in body:
        row_tokens = count_tokens(row)
        if current and current_tokens + row_tokens > TARGET_TOKENS:
            text = "\n".join(header + current)
            pieces.append(_Block(text, block.heading_path, True, count_tokens(text)))
            current = []
            current_tokens = header_tokens
        current.append(row)
        current_tokens += row_tokens

    if current:
        text = "\n".join(header + current)
        pieces.append(_Block(text, block.heading_path, True, count_tokens(text)))

    return pieces or [block]


def _tail_overlap(text: str, tokens: int) -> str:
    """Last ~`tokens` worth of text, cut on a line boundary."""
    lines = text.splitlines()
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        line_tokens = count_tokens(line)
        if total + line_tokens > tokens and kept:
            break
        kept.insert(0, line)
        total += line_tokens
    return "\n".join(kept)


def chunk_markdown(markdown: str) -> list[Chunk]:
    """Split a parsed document into embeddable chunks."""
    blocks: list[_Block] = []
    for block in _split_into_blocks(markdown):
        if block.tokens > TARGET_TOKENS and block.is_table:
            blocks.extend(_split_table(block))
        elif block.tokens > TARGET_TOKENS:
            blocks.extend(_split_prose(block))
        else:
            blocks.append(block)

    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_tokens = 0
    carry = ""

    def emit() -> None:
        nonlocal current, current_tokens, carry
        if not current:
            return
        heading_path = current[0].heading_path
        body = "\n\n".join(b.text for b in current)
        text = f"{carry}\n\n{body}".strip() if carry else body
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                heading_path=heading_path,
                text=text,
                token_count=count_tokens(text),
            )
        )
        # Overlap only inside prose; repeating table rows would duplicate
        # requirement ids across chunks and confuse citation.
        carry = "" if current[-1].is_table else _tail_overlap(body, OVERLAP_TOKENS)
        current = []
        current_tokens = 0

    for block in blocks:
        starts_new_section = bool(
            current and block.heading_path != current[0].heading_path
        )
        too_big = current_tokens + block.tokens > TARGET_TOKENS
        if current and (starts_new_section or too_big):
            emit()
        current.append(block)
        current_tokens += block.tokens

    emit()

    return _merge_tiny(chunks)


def _split_prose(block: _Block) -> list[_Block]:
    """Split an oversized paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", block.text)
    pieces: list[_Block] = []
    current: list[str] = []
    total = 0
    for sentence in sentences:
        tokens = count_tokens(sentence)
        if current and total + tokens > TARGET_TOKENS:
            text = " ".join(current)
            pieces.append(_Block(text, block.heading_path, False, count_tokens(text)))
            current = []
            total = 0
        current.append(sentence)
        total += tokens
    if current:
        text = " ".join(current)
        pieces.append(_Block(text, block.heading_path, False, count_tokens(text)))
    return pieces or [block]


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """Fold sub-threshold chunks into their neighbour and renumber."""
    if not chunks:
        return []

    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.token_count < MIN_CHUNK_TOKENS
            and merged[-1].heading_path == chunk.heading_path
        ):
            previous = merged.pop()
            text = f"{previous.text}\n\n{chunk.text}"
            merged.append(
                Chunk(
                    ordinal=previous.ordinal,
                    heading_path=previous.heading_path,
                    text=text,
                    token_count=count_tokens(text),
                )
            )
        else:
            merged.append(chunk)

    return [
        Chunk(
            ordinal=index,
            heading_path=c.heading_path,
            text=c.text,
            token_count=c.token_count,
        )
        for index, c in enumerate(merged)
    ]
