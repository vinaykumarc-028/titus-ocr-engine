from __future__ import annotations

from app.ocr.models import PageResult, ParsedDocument
from app.ocr.parser import parse_markdown


def analyze_examination_page(page_number: int, ocr_markdown: str) -> PageResult:
    """
    Convert verbatim OCR markdown into TITUS semantic examination elements.

    This is the explicit Document Intelligence stage:
      OCR output -> structured examination model -> review/composition.

    The implementation is deterministic and does not rewrite OCR text. It only
    classifies visible blocks, markers, marks, hierarchy, and question types.
    """
    return parse_markdown(page_number=page_number, markdown=ocr_markdown)


def analyze_examination_document(pages: list[tuple[int, str]]) -> ParsedDocument:
    return ParsedDocument(
        pages=[
            analyze_examination_page(page_number=page_number, ocr_markdown=markdown)
            for page_number, markdown in pages
        ]
    )
