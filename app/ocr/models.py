from enum import StrEnum

from pydantic import BaseModel, Field


class ElementType(StrEnum):
    document_title = "document_title"
    institution_name = "institution_name"
    exam_name = "exam_name"
    section_heading = "section_heading"
    subsection_heading = "subsection_heading"
    instruction = "instruction"
    question_group = "question_group"
    question = "question"
    sub_question = "sub_question"
    mcq_option = "mcq_option"
    true_false = "true_false"
    assertion_reason = "assertion_reason"
    case_study = "case_study"
    mark_allocation = "mark_allocation"
    fill_blank = "fill_blank"
    paragraph = "paragraph"
    table = "table"
    header = "header"
    match_row = "match_row"
    diagram_placeholder = "diagram_placeholder"
    image = "image"
    footer = "footer"
    page_break = "page_break"
    signature = "signature"


class DocumentElement(BaseModel):
    id: str | None = None
    type: ElementType
    text: str = Field(min_length=1)
    line_number: int = Field(ge=1)
    raw_text: str = Field(min_length=1)
    marker: str | None = None
    mark_allocation: str | None = None
    parent_marker: str | None = None
    hierarchy_level: int = Field(default=0, ge=0)
    question_type: str | None = None
    has_fill_blank: bool = False
    match_column_a: str | None = None
    match_column_b: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: list[float] | None = None


class PageResult(BaseModel):
    page: int = Field(ge=1)
    markdown: str
    plain_text: str
    elements: list[DocumentElement]


class ParsedDocument(BaseModel):
    pages: list[PageResult]


class UploadResponse(ParsedDocument):
    pass


class EvaluationResult(BaseModel):
    """Legacy evaluation result - kept for backward compatibility with CLI/tests."""
    character_accuracy: float
    word_accuracy: float
    missing_characters: int
    extra_characters: int
    processing_time_seconds: float
    unified_diff: str


# ---------------------------------------------------------------------------
# Production-grade benchmark models
# ---------------------------------------------------------------------------


class PageEvaluationResult(BaseModel):
    """Per-page benchmarking result from the production OCR evaluation engine."""
    page: int

    # Character-level metrics
    cer: float = Field(ge=0.0, le=1.0)
    character_accuracy: float = Field(ge=0.0, le=1.0)
    chars_reference: int = 0
    chars_correct: int = 0
    chars_substituted: int = 0
    chars_deleted: int = 0
    chars_inserted: int = 0

    # Word-level metrics
    wer: float = Field(ge=0.0, le=1.0)
    word_accuracy: float = Field(ge=0.0, le=1.0)
    words_reference: int = 0
    words_correct: int = 0
    words_substituted: int = 0
    words_deleted: int = 0
    words_inserted: int = 0

    # Production validation metrics
    missing_characters: int = 0
    inserted_characters: int = 0
    deleted_characters: int = 0
    missing_words: int = 0
    inserted_words: int = 0
    deleted_words: int = 0
    formatting_score: float = Field(default=1.0, ge=0.0, le=1.0)
    structural_score: float = Field(default=1.0, ge=0.0, le=1.0)
    page_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)

    # Structural anchors found in this page's ground truth
    anchors_found: list[str] = Field(default_factory=list)
    anchors_missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Side-by-side comparison payloads
    ground_truth_text: str = ""
    ocr_text: str = ""

    # Unified diff (debug only — NOT used for scoring)
    unified_diff: str = ""


class OcrBenchmarkReport(BaseModel):
    """Document-level aggregate OCR benchmark report."""

    # Macro averages (mean over pages)
    avg_cer: float
    avg_wer: float
    avg_character_accuracy: float
    avg_word_accuracy: float

    # Micro totals (summed across all pages)
    total_chars_reference: int
    total_chars_correct: int
    total_chars_substituted: int
    total_chars_deleted: int
    total_chars_inserted: int
    total_words_reference: int
    total_words_correct: int
    total_words_substituted: int
    total_words_deleted: int
    total_words_inserted: int

    # Document-level CER/WER from totals (micro-average — the canonical score)
    document_cer: float
    document_wer: float
    document_character_accuracy: float
    document_word_accuracy: float

    # Confidence distribution (by page char accuracy)
    confidence_high: int    # Pages with char_acc >= 95%
    confidence_medium: int  # Pages with 80% <= char_acc < 95%
    confidence_low: int     # Pages with char_acc < 80%

    # Best and worst pages by CER
    lowest_cer_page: int
    highest_cer_page: int
    lowest_cer: float
    highest_cer: float

    # Per-page breakdown
    pages: list[PageEvaluationResult] = Field(default_factory=list)

    # Metadata extracted from ground truth preamble (for display only)
    ground_truth_metadata: dict[str, str] = Field(default_factory=dict)

    # Production validation totals and preservation scores
    total_missing_characters: int = 0
    total_inserted_characters: int = 0
    total_deleted_characters: int = 0
    total_missing_words: int = 0
    total_inserted_words: int = 0
    total_deleted_words: int = 0
    formatting_preservation: float = Field(default=1.0, ge=0.0, le=1.0)
    structural_preservation: float = Field(default=1.0, ge=0.0, le=1.0)
    overall_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)

    # Processing stats
    processing_time_seconds: float = 0.0

    # Document-level unified diff (debug only)
    unified_diff: str = ""


# ---------------------------------------------------------------------------
# Pipeline models
# ---------------------------------------------------------------------------


class PageMetrics(BaseModel):
    preprocessing_time_seconds: float = 0.0
    image_size_bytes: int = 0
    image_resolution: str = ""
    gemini_request_seconds: float = 0.0
    parsing_time_seconds: float = 0.0
    total_time_seconds: float = 0.0


class OCRValidationPage(BaseModel):
    page: int = Field(ge=1)
    status: str = "Completed"  # e.g. "Completed", "Timeout", "Failed"
    error_reason: str | None = None
    image_data_url: str
    markdown: str = ""
    plain_text: str = ""
    elements: list[DocumentElement] = Field(default_factory=list)
    processing_time_seconds: float = Field(ge=0)
    metrics: PageMetrics | None = None
    retry_count: int = 0
    quality_report: dict | None = None


class OCRRunSummary(BaseModel):
    pages_processed: int
    pages_succeeded: int
    pages_failed: int
    avg_page_processing_time_seconds: float
    total_processing_time_seconds: float
    avg_image_size_bytes: float
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OCRValidationResponse(BaseModel):
    pages: list[OCRValidationPage]
    total_processing_time_seconds: float = Field(ge=0)
    evaluation: EvaluationResult | None = None       # legacy field
    benchmark: OcrBenchmarkReport | None = None       # production benchmark
    run_summary: OCRRunSummary | None = None
