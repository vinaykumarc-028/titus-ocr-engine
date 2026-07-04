from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "titus.exam.v1"


class BlockType(StrEnum):
    header = "header"
    footer = "footer"
    section = "section"
    subsection = "subsection"
    instruction = "instruction"
    question = "question"
    sub_question = "sub_question"
    marks = "marks"
    mcq = "mcq"
    option = "option"
    fill_blank = "fill_blank"
    match_following = "match_following"
    table = "table"
    paragraph = "paragraph"
    diagram_placeholder = "diagram_placeholder"
    image = "image"
    signature = "signature"
    unreadable_marker = "unreadable_marker"
    page_break = "page_break"


class Metadata(BaseModel):
    title: str = "Untitled Examination Document"
    institution_name: str = "TITUS SOLUTIONS EXAM LAB"
    subject: str | None = None
    class_grade: str | None = None
    language: str = "English"
    category: str | None = None
    total_marks: str | None = None
    time_allowed: str | None = None
    instructions: str | None = None
    notes: str | None = None
    source_filenames: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    page_number: int = Field(ge=1)
    line_number: int | None = Field(default=None, ge=1)
    source_element_type: str | None = None
    raw_text: str | None = None
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Marks(BaseModel):
    raw: str
    value: float | None = None
    unit: str = "marks"


class Option(BaseModel):
    id: str
    label: str | None = None
    text: str
    is_correct: bool | None = None


class MatchPair(BaseModel):
    left: str
    right: str


class TableCell(BaseModel):
    text: str
    header: bool = False


class TableRow(BaseModel):
    cells: list[TableCell] = Field(default_factory=list)


class Table(BaseModel):
    rows: list[TableRow] = Field(default_factory=list)


class StructuredBlock(BaseModel):
    id: str
    type: BlockType
    text: str = ""
    marker: str | None = None
    parent_marker: str | None = None
    hierarchy_level: int = Field(default=0, ge=0)
    question_type: str | None = None
    marks: Marks | None = None
    options: list[Option] = Field(default_factory=list)
    match_pairs: list[MatchPair] = Field(default_factory=list)
    table: Table | None = None
    image_url: str | None = None
    source: SourceReference
    metadata: dict[str, Any] = Field(default_factory=dict)


class Header(StructuredBlock):
    type: Literal[BlockType.header] = BlockType.header


class Footer(StructuredBlock):
    type: Literal[BlockType.footer] = BlockType.footer


class Section(StructuredBlock):
    type: Literal[BlockType.section] = BlockType.section


class Subsection(StructuredBlock):
    type: Literal[BlockType.subsection] = BlockType.subsection


class Instruction(StructuredBlock):
    type: Literal[BlockType.instruction] = BlockType.instruction


class Question(StructuredBlock):
    type: Literal[BlockType.question] = BlockType.question


class SubQuestion(StructuredBlock):
    type: Literal[BlockType.sub_question] = BlockType.sub_question


class MCQ(StructuredBlock):
    type: Literal[BlockType.mcq] = BlockType.mcq


class FillBlank(StructuredBlock):
    type: Literal[BlockType.fill_blank] = BlockType.fill_blank


class MatchFollowing(StructuredBlock):
    type: Literal[BlockType.match_following] = BlockType.match_following


class Paragraph(StructuredBlock):
    type: Literal[BlockType.paragraph] = BlockType.paragraph


class DiagramPlaceholder(StructuredBlock):
    type: Literal[BlockType.diagram_placeholder] = BlockType.diagram_placeholder


class Image(StructuredBlock):
    type: Literal[BlockType.image] = BlockType.image


class Signature(StructuredBlock):
    type: Literal[BlockType.signature] = BlockType.signature


class UnreadableMarker(StructuredBlock):
    type: Literal[BlockType.unreadable_marker] = BlockType.unreadable_marker


class PageBreak(StructuredBlock):
    type: Literal[BlockType.page_break] = BlockType.page_break


class Page(BaseModel):
    page_number: int = Field(ge=1)
    status: str = "completed"
    image_url: str | None = None
    blocks: list[StructuredBlock] = Field(default_factory=list)


class Document(BaseModel):
    schema_version: str = SCHEMA_VERSION
    document_id: str
    metadata: Metadata = Field(default_factory=Metadata)
    pages: list[Page] = Field(default_factory=list)


StructuredExaminationDocument = Document
