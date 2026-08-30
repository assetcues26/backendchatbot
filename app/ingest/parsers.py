"""Document parsing: .docx, .xlsx, .pdf, .md, .txt -> structured markdown.

This is a direct descendant of the extractor that was run against all 21
AssetCues product files, so the shapes it handles are the shapes those files
actually contain: Word headings, dense requirement tables (UAP-FR-001 style),
and Excel QA workbooks with a Summary sheet and a wide Test Cases sheet.

Two things it does that a naive text dump does not:

  - Tables become markdown tables. The AssetCues specs put nearly all their
    substance in tables -- requirement id, rule, acceptance criterion -- and
    flattening those to prose destroys the row association that makes an
    answer correct.
  - Headings are preserved as markdown headings, which the chunker then uses
    as split points and as the `heading_path` shown in citations.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: S405 - types only, parsing uses DefusedET

from defusedxml import ElementTree as DefusedET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

SUPPORTED_SUFFIXES = {".docx", ".xlsx", ".pdf", ".md", ".txt"}

# The AssetCues document template carries an explicit audience field. It is a
# human-authored classification signal and we would be foolish to ignore it.
_AUDIENCE_ROW = re.compile(
    r"\|\s*[^|]*?\|\s*((?:[A-Z][A-Za-z&/ ]+)(?:,\s*[A-Za-z&/ ]+)*(?:\s+and\s+[A-Za-z&/ ]+)?)\s*\|",
)
_AUDIENCE_HEADER = re.compile(r"\|\s*Document purpose\s*\|\s*Primary audience\s*\|", re.I)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    markdown: str
    declared_audience: list[str] = field(default_factory=list)
    doc_type: str = ""
    page_count: int = 0

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())


class UnsupportedDocument(Exception):
    pass


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------


def _para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(W + "t"))


def _para_style(p: ET.Element) -> str:
    p_pr = p.find(W + "pPr")
    if p_pr is None:
        return ""
    style = p_pr.find(W + "pStyle")
    return style.get(W + "val", "") if style is not None else ""


def parse_docx(path: Path) -> ParsedDocument:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")

    body = DefusedET.fromstring(xml).find(W + "body")
    if body is None:
        raise UnsupportedDocument(f"{path.name}: no document body")

    lines: list[str] = []
    for element in body:
        tag = element.tag.replace(W, "")

        if tag == "p":
            text = _para_text(element).strip()
            if not text:
                continue
            style = _para_style(element).lower()
            if style.startswith("heading"):
                level = int(re.sub(r"\D", "", style) or "1")
                lines.append("#" * min(level, 6) + " " + text)
            elif style == "title":
                lines.append("# " + text)
            else:
                lines.append(text)

        elif tag == "tbl":
            rows: list[list[str]] = []
            for tr in element.findall(W + "tr"):
                cells = [
                    " ".join(_para_text(p).strip() for p in tc.findall(W + "p")).strip()
                    for tc in tr.findall(W + "tc")
                ]
                rows.append(cells)
            if rows:
                lines.append("")
                width = max(len(r) for r in rows)
                for index, row in enumerate(rows):
                    padded = row + [""] * (width - len(row))
                    lines.append("| " + " | ".join(padded) + " |")
                    if index == 0:
                        lines.append("|" + "---|" * width)
                lines.append("")

    markdown = "\n".join(lines)
    return ParsedDocument(
        title=_derive_title(lines, path),
        markdown=markdown,
        declared_audience=extract_declared_audience(markdown),
        doc_type=_derive_doc_type(path.name, markdown),
    )


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def parse_xlsx(path: Path) -> ParsedDocument:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = DefusedET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(S + "t"))
                for si in root.findall(S + "si")
            ]

        workbook = DefusedET.fromstring(archive.read("xl/workbook.xml"))
        sheets = [(s.get("name", ""), s.get(R + "id", "")) for s in workbook.iter(S + "sheet")]

        rels = {
            rel.get("Id", ""): rel.get("Target", "")
            for rel in DefusedET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }

        lines: list[str] = []
        for sheet_name, rel_id in sheets:
            target = rels.get(rel_id, "").lstrip("/").removeprefix("xl/")
            member = f"xl/{target}"
            if member not in names:
                continue

            lines.append(f"## Sheet: {sheet_name}")
            sheet = DefusedET.fromstring(archive.read(member))
            for row in sheet.iter(S + "row"):
                values: list[str] = []
                for cell in row.findall(S + "c"):
                    v = cell.find(S + "v")
                    inline = cell.find(S + "is")
                    if cell.get("t") == "s" and v is not None and v.text:
                        index = int(v.text)
                        values.append(shared[index] if index < len(shared) else "")
                    elif inline is not None:
                        values.append("".join(t.text or "" for t in inline.iter(S + "t")))
                    elif v is not None:
                        values.append(v.text or "")
                    else:
                        values.append("")
                if any(value.strip() for value in values):
                    lines.append("| " + " | ".join(values) + " |")
            lines.append("")

    markdown = "\n".join(lines)
    return ParsedDocument(
        # A workbook has no H1 to borrow, so the cleaned filename is the title.
        title=_clean_filename_title(path),
        markdown=markdown,
        declared_audience=[],
        doc_type=_derive_doc_type(path.name, markdown),
    )


# ---------------------------------------------------------------------------
# PDF and plain text
# ---------------------------------------------------------------------------


def parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    markdown = "\n\n".join(p for p in pages if p)
    return ParsedDocument(
        title=_clean_filename_title(path),
        markdown=markdown,
        declared_audience=extract_declared_audience(markdown),
        doc_type=_derive_doc_type(path.name, markdown),
        page_count=len(reader.pages),
    )


def parse_text(path: Path) -> ParsedDocument:
    markdown = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDocument(
        title=_clean_filename_title(path),
        markdown=markdown,
        declared_audience=extract_declared_audience(markdown),
        doc_type=_derive_doc_type(path.name, markdown),
    )


# ---------------------------------------------------------------------------
# Dispatch and heuristics
# ---------------------------------------------------------------------------


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return parse_docx(path)
    if suffix == ".xlsx":
        return parse_xlsx(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".md", ".txt"}:
        return parse_text(path)
    raise UnsupportedDocument(
        f"{path.name}: unsupported type {suffix!r} "
        f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
    )


def extract_declared_audience(markdown: str) -> list[str]:
    """Pull the 'Primary audience' cell out of the AssetCues doc template.

    Returns the audience names as written, e.g.
    ["Product", "Engineering", "QA", "Implementation", "Support", "Audit"].
    The classifier gets these as evidence; it still has to map them onto our
    role keys, because the document names job functions, not our roles.
    """
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not _AUDIENCE_HEADER.search(line):
            continue
        # Header row, separator row, then the values row.
        for candidate in lines[index + 1 : index + 4]:
            cells = [c.strip() for c in candidate.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] and not set(cells[1]) <= {"-"}:
                return _split_audience(cells[1])
    return []


def _split_audience(value: str) -> list[str]:
    """Split an audience cell into names.

    Split on commas, then split only the FINAL element on " and ". That is
    what these documents actually look like:

        "Product, Engineering, QA, Support, Security and Audit"
            -> [..., "Security", "Audit"]                       correct
        "Group and Legal Entity Administrators, Approvers, ..."
            -> ["Group and Legal Entity Administrators", ...]   left intact

    Splitting every " and " would shred that first entry into two role names
    that do not exist.
    """
    parts = [p.strip(" .") for p in value.split(",") if p.strip(" .")]
    if parts:
        tail = re.split(r"\s+and\s+", parts[-1])
        parts = parts[:-1] + [t.strip(" .") for t in tail if t.strip(" .")]
    return [p for p in parts if 0 < len(p) < 60]


# H1s that are structural furniture rather than the document's name.
_TITLE_STOPLIST = {
    "contents",
    "document control",
    "about this guide",
    "executive summary",
    "introduction",
    "table of contents",
    "quick reference",
    "overview",
    "purpose",
    "scope",
    "glossary",
    "definitions",
    "revision history",
    "document history",
    "how to use this guide",
    "capability at a glance",
}


def _clean_filename_title(path: Path) -> str:
    """`01_OSM_Lean_Product_and_Functional_Specification (3).docx`
    -> `OSM Lean Product and Functional Specification`."""
    stem = path.stem
    stem = re.sub(r"^\d{1,2}[_\-\s]+", "", stem)  # leading 01_
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)  # Drive's " (3)" suffix
    stem = re.sub(r"[_\-\s]*v\d+(\.\d+)*$", "", stem, flags=re.I)  # _v1.1
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s{2,}", " ", stem).strip(" -")
    return stem[:500]


def _derive_title(lines: list[str], path: Path) -> str:
    """Prefer the document's own H1, but only if it names the document.

    Several of these files style their cover page as plain paragraphs, so the
    first H1 is a section heading like "1. Capability at a glance". Using that
    as the title makes both citations and the admin list unreadable.
    """
    for line in lines[:60]:
        if not line.startswith("# "):
            continue
        candidate = line[2:].strip()
        if re.match(r"^\d+[.)]", candidate):  # "1. Capability at a glance"
            continue
        if candidate.lower().strip(" .:") in _TITLE_STOPLIST:
            continue
        if len(candidate) < 3:
            continue
        return candidate[:500]
    return _clean_filename_title(path)


# Filename is checked before content. These files are named systematically,
# whereas body text says "validation" and "requirements" constantly -- content
# matching alone mislabels most of the specifications.
_DOC_TYPE_RULES: list[tuple[str, str]] = [
    (r"test[\s_-]*cases?", "Test Cases"),
    (r"validation|governance|traceability", "Validation & Governance Pack"),
    (r"functional[\s_-]*specification|product[\s_-]*and[\s_-]*functional",
     "Product & Functional Specification"),
    (r"\bbrd\b|business[\s_-]*requirements?", "Business Requirements Document"),
    (r"user[\s_-]*manual", "User Manual"),
    (r"administrator[\s_-]*guide|admin[\s_-]*guide",
     "User & Administrator Guide"),
]


def _derive_doc_type(filename: str, markdown: str) -> str:
    """Best-effort label. The classifier may override it."""
    # Underscores are word characters, so "_brd_" defeats . Normalise
    # separators to spaces first.
    name = re.sub(r"[_\-]+", " ", filename.lower())
    for pattern, label in _DOC_TYPE_RULES:
        if re.search(pattern, name):
            return label
    # Fall back to the masthead only -- the first few lines, where the
    # template prints the document kind -- not the whole body.
    head = "\n".join(markdown.splitlines()[:12]).lower()
    for pattern, label in _DOC_TYPE_RULES:
        if re.search(pattern, head):
            return label
    return "Document"
