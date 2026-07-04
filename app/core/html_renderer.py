from __future__ import annotations

import html
import re
from datetime import date

from app.core.examination_model import BlockType, Document, Metadata, StructuredBlock
from app.core.question_paper_composer import ComposedPage, QuestionPaperComposer


class HTMLRenderer:
    format_name = "html"
    file_extension = "html"
    media_type = "text/html; charset=utf-8"
    production_enabled = True

    def __init__(self, composer: QuestionPaperComposer | None = None) -> None:
        self.composer = composer or QuestionPaperComposer()

    def render(self, document: Document) -> str:
        paper = self.composer.compose(document)
        metadata = paper.document.metadata
        title = metadata.title or metadata.subject or "TITUS Examination Document"
        pages_html = "\n".join(
            self._render_page(metadata, page, len(paper.pages))
            for page in paper.pages
        )
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{_escape(title)}</title>",
                "<style>",
                _css(),
                "</style>",
                "</head>",
                "<body>",
                '<main class="titus-document" aria-label="Structured examination document">',
                pages_html or self._render_empty_page(metadata),
                "</main>",
                "</body>",
                "</html>",
            ]
        )

    def render_bytes(self, document: Document) -> bytes:
        return self.render(document).encode("utf-8")

    def _render_page(self, metadata: Metadata, page: ComposedPage, total_pages: int) -> str:
        blocks = "\n".join(self._render_block(block) for block in page.blocks)
        if not blocks.strip():
            blocks = '<p class="unreadable">No structured text was recovered for this page.</p>'
        return f"""
<article class="page" data-page="{page.page_number}">
  {self._render_header(metadata)}
  <section class="page-body">
    {blocks}
  </section>
  <footer class="page-footer">
    <span>{_escape(metadata.institution_name)}</span>
    <span>Page {page.page_number} of {total_pages}</span>
    <span>{_escape(metadata.subject or "")}</span>
  </footer>
</article>
""".strip()

    def _render_empty_page(self, metadata: Metadata) -> str:
        return f"""
<article class="page" data-page="1">
  {self._render_header(metadata)}
  <section class="page-body">
    <p class="unreadable">No structured pages are available.</p>
  </section>
  <footer class="page-footer"><span></span><span>Page 1 of 1</span><span></span></footer>
</article>
""".strip()

    def _render_header(self, metadata: Metadata) -> str:
        detail_items = [
            ("Subject", metadata.subject),
            ("Class", metadata.class_grade),
            ("Time", metadata.time_allowed),
            ("Marks", metadata.total_marks),
        ]
        details = "".join(
            f"<span><strong>{_escape(label)}:</strong> {_escape(value)}</span>"
            for label, value in detail_items
            if value
        )
        instructions = (
            f'<p class="document-instructions">{_inline(metadata.instructions)}</p>'
            if metadata.instructions
            else ""
        )
        notes = f'<p class="document-notes">{_inline(metadata.notes)}</p>' if metadata.notes else ""
        return f"""
<header class="doc-header">
  <div class="brand-mark">TITUS</div>
  <div class="header-main">
    <h1>{_escape(metadata.institution_name)}</h1>
    <h2>{_escape(metadata.title)}</h2>
    <div class="header-meta">{details}</div>
    {instructions}
    {notes}
  </div>
</header>
""".strip()

    def _render_block(self, block: StructuredBlock) -> str:
        if block.type == BlockType.header:
            return f'<p class="running-header">{_line(block)}</p>'
        if block.type == BlockType.footer:
            return f'<p class="running-footer">{_line(block)}</p>'
        if block.type == BlockType.section:
            return f'<section class="section"><h3>{_line(block)}</h3></section>'
        if block.type == BlockType.subsection:
            return f'<section class="subsection"><h4>{_line(block)}</h4></section>'
        if block.type == BlockType.instruction:
            return f'<p class="instruction">{_line(block)}</p>'
        if block.type == BlockType.question:
            return _question("question", block)
        if block.type == BlockType.mcq:
            return _question("question mcq-question", block)
        if block.type == BlockType.sub_question:
            return _question("sub-question", block)
        if block.type == BlockType.option:
            return _option(block)
        if block.type == BlockType.fill_blank:
            return _question("fill-blank", block)
        if block.type == BlockType.match_following:
            return _match_table(block)
        if block.type == BlockType.table:
            return _table(block)
        if block.type == BlockType.diagram_placeholder:
            return _diagram(block)
        if block.type == BlockType.image:
            return _image(block)
        if block.type == BlockType.signature:
            return f'<div class="signature-line">{_line(block)}</div>'
        if block.type == BlockType.page_break:
            return '<div class="explicit-page-break" aria-label="Page break"></div>'
        if block.type == BlockType.unreadable_marker:
            return f'<p class="unreadable">{_line(block)}</p>'
        if block.type == BlockType.marks:
            return f'<p class="marks-line">{_line(block)}</p>'
        return f'<p class="paragraph">{_line(block)}</p>'


def html_filename(metadata: Metadata) -> str:
    parts = [
        metadata.subject or metadata.title or "TITUS_Document",
        metadata.class_grade,
        date.today().isoformat(),
    ]
    return f"{'_'.join(_slug(part) for part in parts if part)}.html"


def _clean_text_with_marker(text: str | None, marker: str | None) -> str:
    if not text:
        return ""
    if not marker:
        return text
    txt_stripped = text.strip()
    m_stripped = marker.strip()
    if not m_stripped:
        return text

    if txt_stripped.lower().startswith(m_stripped.lower()):
        rest = txt_stripped[len(m_stripped):].lstrip()
        if rest.startswith(('.', ')', '-', ' ')):
            if not m_stripped.endswith(rest[0]):
                rest = rest[1:].lstrip()
        return rest
    return text


def _question(css_class: str, block: StructuredBlock) -> str:
    marker = f'<span class="marker">{_escape(block.marker)}</span>' if block.marker else ""
    marks = _marks(block)
    cleaned_text = _clean_text_with_marker(block.text, block.marker)
    return f"""
<article class="{css_class}" data-block-id="{_escape_attr(block.id)}">
  <span class="question-line">{marker}<span>{_inline(cleaned_text)}</span></span>
  {marks}
</article>
""".strip()


def _option(block: StructuredBlock) -> str:
    label = block.marker or (block.options[0].label if block.options else "")
    marker = f'<span class="option-label">{_escape(label)}</span>' if label else ""
    cleaned_text = _clean_text_with_marker(block.text, label)
    return f'<p class="option" data-block-id="{_escape_attr(block.id)}">{marker}<span>{_inline(cleaned_text)}</span></p>'


def _match_table(block: StructuredBlock) -> str:
    rows = "\n".join(
        f"<tr><td>{_inline(pair.left)}</td><td>{_inline(pair.right)}</td></tr>"
        for pair in block.match_pairs
    )
    return f"""
<table class="match-table" data-block-id="{_escape_attr(block.id)}">
  <thead><tr><th>Column A</th><th>Column B</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
""".strip()


def _table(block: StructuredBlock) -> str:
    if not block.table:
        return f'<p class="paragraph">{_line(block)}</p>'
    rows = []
    for row in block.table.rows:
        cells = "".join(
            f"<{'th' if cell.header else 'td'}>{_inline(cell.text)}</{'th' if cell.header else 'td'}>"
            for cell in row.cells
        )
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="data-table" data-block-id="{_escape_attr(block.id)}"><tbody>{"".join(rows)}</tbody></table>'


def _diagram(block: StructuredBlock) -> str:
    return f"""
<figure class="diagram-placeholder" data-block-id="{_escape_attr(block.id)}">
  <figcaption>{_line(block)}</figcaption>
  <div>Diagram space</div>
</figure>
""".strip()


def _image(block: StructuredBlock) -> str:
    if not block.image_url:
        return _diagram(block)
    return f"""
<figure class="embedded-image" data-block-id="{_escape_attr(block.id)}">
  <img src="{_escape_attr(block.image_url)}" alt="{_escape_attr(block.text or "Document image")}">
  <figcaption>{_inline(block.text)}</figcaption>
</figure>
""".strip()


def _line(block: StructuredBlock) -> str:
    return _inline(block.text)


def _marks(block: StructuredBlock) -> str:
    if not block.marks:
        return ""
    return f'<strong class="marks">{_escape(block.marks.raw)}</strong>'


def _inline(value: str | None) -> str:
    escaped = _escape(value or "")
    escaped = re.sub(
        r"&lt;span\s+(?:class=[\"']low-confidence[\"']\s+data-confidence=[\"'](\d+)[\"']|data-confidence=[\"'](\d+)[\"']\s+class=[\"']low-confidence[\"'])\s*&gt;(.*?)&lt;/span&gt;",
        lambda m: f'<span class="low-confidence" data-confidence="{m.group(1) or m.group(2)}" title="Confidence: {m.group(1) or m.group(2)}%">{m.group(3)}</span>',
        escaped,
        flags=re.IGNORECASE
    )
    escaped = re.sub(
        r"(\[UNREADABLE[^\]]*\])",
        r'<span class="unreadable-inline">\1</span>',
        escaped,
        flags=re.IGNORECASE,
    )
    escaped = re.sub(r"_{3,}|-{3,}|\.{3,}", '<span class="blank"></span>', escaped)
    return escaped


def _escape(value: str | None) -> str:
    return html.escape(value or "", quote=False)


def _escape_attr(value: str | None) -> str:
    return html.escape(value or "", quote=True)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "Document"


def _css() -> str:
    return """
@page {
  size: A4 portrait;
  margin: 18mm 16mm 20mm;
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
  background: #eef2f7;
  color: #111827;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 11.5pt;
  line-height: 1.35;
}

.titus-document {
  width: 100%;
}

.page {
  position: relative;
  width: 210mm;
  min-height: 297mm;
  margin: 12mm auto;
  padding: 18mm 16mm 22mm;
  background: #fff;
  box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18);
  page-break-after: always;
  break-after: page;
}

.page:last-child {
  page-break-after: auto;
  break-after: auto;
}

.doc-header {
  display: grid;
  grid-template-columns: 24mm 1fr;
  gap: 8mm;
  align-items: start;
  padding-bottom: 6mm;
  border-bottom: 1.5pt solid #0f3f76;
  margin-bottom: 7mm;
  break-inside: avoid;
}

.brand-mark {
  width: 24mm;
  height: 24mm;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1pt solid #0f3f76;
  color: #0f3f76;
  font-weight: 700;
  font-size: 10pt;
  letter-spacing: 0;
}

.header-main {
  text-align: center;
}

.header-main h1,
.header-main h2 {
  margin: 0;
  line-height: 1.18;
}

.header-main h1 {
  font-size: 15pt;
  font-weight: 700;
  text-transform: uppercase;
}

.header-main h2 {
  margin-top: 2mm;
  font-size: 13pt;
  font-weight: 600;
}

.header-meta {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 2mm 8mm;
  margin-top: 4mm;
  font-size: 10.5pt;
}

.document-instructions,
.document-notes {
  margin: 3mm 0 0;
  font-size: 10pt;
  color: #374151;
}

.page-body {
  padding-bottom: 16mm;
}

.running-header,
.running-footer {
  text-align: center;
  font-size: 10.5pt;
  color: #475569;
}

.section {
  margin: 7mm 0 5mm;
  padding: 2mm 0;
  border-top: 0.75pt solid #cbd5e1;
  border-bottom: 0.75pt solid #cbd5e1;
  text-align: center;
  break-inside: avoid;
}

.section h3,
.subsection h4 {
  margin: 0;
  font-size: 12.5pt;
  font-weight: 700;
}

.subsection {
  margin: 5mm 0 3mm;
  break-inside: avoid;
}

p,
article,
figure,
table {
  margin: 0 0 3.2mm;
}

.instruction {
  padding-left: 5mm;
  color: #334155;
  font-style: italic;
  break-inside: avoid;
}

.question,
.sub-question,
.fill-blank {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 7mm;
  break-inside: avoid;
}

.sub-question {
  margin-left: 8mm;
}

.question-line {
  display: inline-flex;
  gap: 2.5mm;
  min-width: 0;
}

.marker,
.option-label {
  font-weight: 700;
  white-space: nowrap;
}

.marks {
  white-space: nowrap;
  font-weight: 700;
}

.option {
  display: flex;
  gap: 3mm;
  margin-left: 16mm;
}

.blank {
  display: inline-block;
  min-width: 30mm;
  border-bottom: 0.75pt solid #111827;
  transform: translateY(-1pt);
}

.match-table,
.data-table {
  width: 100%;
  border-collapse: collapse;
  break-inside: avoid;
}

.match-table th,
.match-table td,
.data-table th,
.data-table td {
  border: 0.75pt solid #94a3b8;
  padding: 2.5mm 3mm;
  vertical-align: top;
}

.match-table th,
.data-table th {
  background: #f1f5f9;
  text-align: left;
}

.diagram-placeholder {
  border: 0.75pt dashed #94a3b8;
  padding: 4mm;
  min-height: 32mm;
  break-inside: avoid;
}

.diagram-placeholder figcaption {
  font-weight: 600;
  margin-bottom: 4mm;
}

.diagram-placeholder div {
  color: #64748b;
  font-size: 10pt;
}

.embedded-image img {
  max-width: 100%;
  height: auto;
}

.signature-line {
  margin-top: 12mm;
  padding-top: 6mm;
  border-top: 0.75pt solid #111827;
  width: 60mm;
  margin-left: auto;
  text-align: center;
}

.unreadable,
.unreadable-inline {
  color: #b91c1c;
  font-weight: 700;
}

.low-confidence {
  border-bottom: 1.5pt dashed #f59e0b;
  background-color: rgba(245, 158, 11, 0.08);
  cursor: help;
}

.explicit-page-break {
  page-break-after: always;
  break-after: page;
}

.page-footer {
  position: absolute;
  left: 16mm;
  right: 16mm;
  bottom: 10mm;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 6mm;
  align-items: center;
  border-top: 0.75pt solid #cbd5e1;
  padding-top: 2.5mm;
  color: #475569;
  font-size: 9.5pt;
}

.page-footer span:last-child {
  text-align: right;
}

@media print {
  html,
  body {
    background: #fff;
  }

  .page {
    width: auto;
    min-height: auto;
    margin: 0;
    padding: 0;
    box-shadow: none;
  }

  .page-footer {
    left: 0;
    right: 0;
    bottom: 0;
  }
}
""".strip()
