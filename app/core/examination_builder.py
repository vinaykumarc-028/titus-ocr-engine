from __future__ import annotations

import re
from typing import Any

from app.core.examination_model import (
    BlockType,
    Document,
    Marks,
    MatchPair,
    Metadata,
    Option,
    Page,
    SourceReference,
    StructuredBlock,
)
from app.ocr.models import DocumentElement, ElementType, PageResult


MARK_VALUE_RE = re.compile(r"\d+(?:\.\d+)?")


def page_result_to_structured_page(
    page_result: PageResult,
    *,
    status: str = "completed",
    image_url: str | None = None,
) -> Page:
    return elements_to_structured_page(
        page_number=page_result.page,
        elements=page_result.elements,
        status=status,
        image_url=image_url,
    )


def elements_to_structured_page(
    *,
    page_number: int,
    elements: list[DocumentElement],
    status: str = "completed",
    image_url: str | None = None,
) -> Page:
    blocks: list[StructuredBlock] = []
    match_pairs: list[MatchPair] = []
    first_match_element: DocumentElement | None = None

    def flush_match_following() -> None:
        nonlocal match_pairs, first_match_element
        if not match_pairs or first_match_element is None:
            return
        blocks.append(
            StructuredBlock(
                id=_block_id(page_number, first_match_element, len(blocks), "match"),
                type=BlockType.match_following,
                text="Match the following",
                match_pairs=match_pairs,
                source=_source(page_number, first_match_element),
            )
        )
        match_pairs = []
        first_match_element = None

    for index, element in enumerate(elements):
        if element.type == ElementType.match_row:
            if first_match_element is None:
                first_match_element = element
            match_pairs.append(
                MatchPair(
                    left=element.match_column_a or element.text,
                    right=element.match_column_b or "",
                )
            )
            continue

        flush_match_following()
        blocks.append(_element_to_block(page_number, element, index))

    flush_match_following()
    return Page(page_number=page_number, status=status, image_url=image_url, blocks=blocks)


def structured_document_from_job(job: dict[str, Any]) -> Document:
    existing = job.get("structured_document")
    if existing:
        try:
            document = Document.model_validate(existing)
            if document.pages:
                return _ensure_all_job_pages(document, job)
        except Exception:
            pass

    pages: list[Page] = []
    for page in sorted(job.get("pages", []), key=lambda item: item.get("page_number", 0)):
        structured_page = page.get("structured_page")
        if structured_page:
            try:
                pages.append(Page.model_validate(structured_page))
                continue
            except Exception:
                pass

        elements = [DocumentElement(**element) for element in page.get("elements", [])]
        pages.append(
            elements_to_structured_page(
                page_number=int(page.get("page_number") or len(pages) + 1),
                elements=elements,
                status=page.get("status", "completed"),
                image_url=page.get("image_url"),
            )
        )

    return Document(
        document_id=job.get("id", "unknown"),
        metadata=_metadata_from_job(job),
        pages=pages,
    )


def persist_structured_document(job: dict[str, Any]) -> Document:
    document = structured_document_from_job({**job, "structured_document": None})
    job["structured_document"] = document.model_dump()
    structured_by_page = {page.page_number: page for page in document.pages}
    for page in job.get("pages", []):
        page_number = int(page.get("page_number") or 0)
        structured_page = structured_by_page.get(page_number)
        if structured_page:
            page["structured_page"] = structured_page.model_dump()
    return document


def page_plain_text(page: Page) -> str:
    return "\n".join(block.text for block in page.blocks if block.text).strip()


def document_plain_text(document: Document) -> str:
    return "\n\n".join(page_plain_text(page) for page in document.pages).strip()


def _ensure_all_job_pages(document: Document, job: dict[str, Any]) -> Document:
    existing_numbers = {page.page_number for page in document.pages}
    additional_pages = []
    for page in sorted(job.get("pages", []), key=lambda item: item.get("page_number", 0)):
        page_number = int(page.get("page_number") or 0)
        if page_number in existing_numbers:
            continue
        elements = [DocumentElement(**element) for element in page.get("elements", [])]
        additional_pages.append(
            elements_to_structured_page(
                page_number=page_number,
                elements=elements,
                status=page.get("status", "completed"),
                image_url=page.get("image_url"),
            )
        )
    if not additional_pages:
        return document
    return document.model_copy(update={"pages": sorted(document.pages + additional_pages, key=lambda p: p.page_number)})


def _metadata_from_job(job: dict[str, Any]) -> Metadata:
    meta = job.get("metadata", {}) or {}
    source_filenames = []
    for page in job.get("pages", []):
        source = page.get("source_filename")
        if source and source not in source_filenames:
            source_filenames.append(source)

    return Metadata(
        title=meta.get("title") or job.get("name") or "Untitled Examination Document",
        institution_name=meta.get("institutionName") or "TITUS SOLUTIONS EXAM LAB",
        subject=meta.get("subject") or None,
        class_grade=meta.get("classGrade") or None,
        language=meta.get("language") or "English",
        category=meta.get("category") or None,
        total_marks=meta.get("maxMarks") or None,
        time_allowed=meta.get("timeDuration") or None,
        instructions=meta.get("instructions") or None,
        notes=meta.get("notes") or None,
        source_filenames=source_filenames,
    )


def _element_to_block(page_number: int, element: DocumentElement, index: int) -> StructuredBlock:
    block_type = _block_type_from_element(element)
    option = None
    if block_type == BlockType.option:
        option = Option(
            id=_block_id(page_number, element, index, "option"),
            label=element.marker,
            text=element.text,
        )

    return StructuredBlock(
        id=element.id or _block_id(page_number, element, index),
        type=block_type,
        text=element.text,
        marker=element.marker,
        parent_marker=element.parent_marker,
        hierarchy_level=element.hierarchy_level,
        question_type=element.question_type,
        marks=_parse_marks(element.mark_allocation),
        options=[option] if option else [],
        source=_source(page_number, element),
        metadata={
            "has_fill_blank": element.has_fill_blank,
            "source_element_type": str(element.type),
        },
    )


def _block_type_from_element(element: DocumentElement) -> BlockType:
    if "[UNREADABLE" in element.text.upper():
        return BlockType.unreadable_marker

    if element.type in {ElementType.document_title, ElementType.institution_name, ElementType.exam_name, ElementType.header}:
        return BlockType.header
    if element.type == ElementType.footer:
        return BlockType.footer
    if element.type in {ElementType.section_heading, ElementType.question_group}:
        return BlockType.section
    if element.type == ElementType.subsection_heading:
        return BlockType.subsection
    if element.type == ElementType.instruction:
        return BlockType.instruction
    if element.type == ElementType.fill_blank or element.has_fill_blank:
        return BlockType.fill_blank
    if element.type == ElementType.question:
        return BlockType.mcq if element.question_type == "mcq" else BlockType.question
    if element.type in {ElementType.sub_question, ElementType.true_false, ElementType.assertion_reason, ElementType.case_study}:
        return BlockType.sub_question
    if element.type == ElementType.mcq_option:
        return BlockType.option
    if element.type == ElementType.table:
        return BlockType.table
    if element.type == ElementType.mark_allocation:
        return BlockType.marks
    if element.type == ElementType.diagram_placeholder:
        return BlockType.diagram_placeholder
    if element.type == ElementType.image:
        return BlockType.image
    if element.type == ElementType.signature:
        return BlockType.signature
    if element.type == ElementType.page_break:
        return BlockType.page_break
    return BlockType.paragraph


def _source(page_number: int, element: DocumentElement) -> SourceReference:
    return SourceReference(
        page_number=page_number,
        line_number=element.line_number,
        source_element_type=str(element.type),
        raw_text=element.raw_text,
        ocr_confidence=element.confidence,
    )


def _parse_marks(raw: str | None) -> Marks | None:
    if not raw:
        return None
    value_match = MARK_VALUE_RE.search(raw)
    value = float(value_match.group(0)) if value_match else None
    return Marks(raw=raw, value=value)


def _block_id(page_number: int, element: DocumentElement, index: int, suffix: str | None = None) -> str:
    parts = [
        f"p{page_number}",
        f"l{element.line_number}",
        str(index + 1),
        str(element.type).replace("_", "-"),
    ]
    if suffix:
        parts.append(suffix)
    return "-".join(parts)
