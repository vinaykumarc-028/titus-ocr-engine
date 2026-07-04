from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.examination_model import BlockType, Document, Page, StructuredBlock


class ComposedPage(BaseModel):
    page_number: int
    status: str = "completed"
    image_url: str | None = None
    blocks: list[StructuredBlock] = Field(default_factory=list)


class ComposedQuestionPaper(BaseModel):
    document: Document
    pages: list[ComposedPage] = Field(default_factory=list)


class QuestionPaperComposer:
    """
    Converts the canonical Structured Examination Model into a publication
    sequence for renderers. It never reads raw OCR markdown or HTML.
    """

    def compose(self, document: Document) -> ComposedQuestionPaper:
        pages = [
            ComposedPage(
                page_number=page.page_number,
                status=page.status,
                image_url=page.image_url,
                blocks=self._compose_blocks(page),
            )
            for page in sorted(document.pages, key=lambda item: item.page_number)
        ]
        return ComposedQuestionPaper(document=document, pages=pages)

    def _compose_blocks(self, page: Page) -> list[StructuredBlock]:
        composed: list[StructuredBlock] = []
        pending_options: list[StructuredBlock] = []

        def flush_options() -> None:
            nonlocal pending_options
            if not pending_options:
                return
            composed.extend(pending_options)
            pending_options = []

        for block in page.blocks:
            if block.type == BlockType.option:
                pending_options.append(block)
                continue

            flush_options()
            composed.append(self._normalize_block(block))

        flush_options()
        return composed

    def _normalize_block(self, block: StructuredBlock) -> StructuredBlock:
        if block.type == BlockType.paragraph and block.metadata.get("has_fill_blank"):
            return block.model_copy(update={"type": BlockType.fill_blank})
        return block
