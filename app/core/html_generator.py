from __future__ import annotations

from dataclasses import dataclass

from app.core.examination_builder import elements_to_structured_page
from app.core.examination_model import Document, Metadata
from app.core.html_renderer import HTMLRenderer, html_filename
from app.ocr.models import DocumentElement


@dataclass(frozen=True)
class TemplateConfig:
    institution_name: str = "TITUS SOLUTIONS EXAM LAB"
    subject: str | None = None
    class_grade: str | None = None
    total_marks: int | str | None = None
    time_allowed: str | None = None
    notes: str | None = None
    title: str | None = None


def create_html_from_elements(
    page_elements: list[tuple[int, list[DocumentElement]]],
    institution_name: str = "TITUS SOLUTIONS EXAM LAB",
    subject: str | None = None,
    class_grade: str | None = None,
    total_marks: int | str | None = None,
    time_allowed: str | None = None,
    notes: str | None = None,
    title: str | None = None,
) -> str:
    metadata = Metadata(
        title=title or subject or "TITUS Examination Document",
        institution_name=institution_name,
        subject=subject,
        class_grade=class_grade,
        total_marks=str(total_marks) if total_marks is not None else None,
        time_allowed=time_allowed,
        notes=notes,
    )
    pages = [
        elements_to_structured_page(page_number=page_number, elements=elements)
        for page_number, elements in page_elements
    ]
    document = Document(document_id="compatibility-render", metadata=metadata, pages=pages)
    return HTMLRenderer().render(document)


def render_html_document(document: Document) -> str:
    return HTMLRenderer().render(document)
