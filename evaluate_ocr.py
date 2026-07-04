"""
Production-grade OCR benchmarking engine for TITUS-082.

This module owns Validation Lab scoring. It deliberately separates human
ground-truth material from benchmark ground truth before any OCR comparison:

  Ground Truth
    -> metadata / appendix removal
    -> normalization
    -> page extraction
    -> paragraph / line / word / character segmentation
    -> independent page alignment
    -> metrics

Unified diff is retained only as a debugging view. Accuracy is computed from
aligned edit operations, never from the rendered diff.
"""

from __future__ import annotations

import difflib
import html
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Public Data Structures
# ---------------------------------------------------------------------------


@dataclass
class PageEvaluationResult:
    """Per-page benchmarking result."""

    page: int

    # Character-level metrics
    cer: float
    character_accuracy: float
    chars_reference: int
    chars_correct: int
    chars_substituted: int
    chars_deleted: int
    chars_inserted: int

    # Word-level metrics
    wer: float
    word_accuracy: float
    words_reference: int
    words_correct: int
    words_substituted: int
    words_deleted: int
    words_inserted: int

    # Production validation metrics
    missing_characters: int = 0
    inserted_characters: int = 0
    deleted_characters: int = 0
    missing_words: int = 0
    inserted_words: int = 0
    deleted_words: int = 0
    formatting_score: float = 1.0
    structural_score: float = 1.0
    page_accuracy: float = 1.0

    # Structural
    anchors_found: list[str] = field(default_factory=list)
    anchors_missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Side-by-side comparison payloads
    ground_truth_text: str = ""
    ocr_text: str = ""

    # Unified diff (debug only, NOT used for scoring)
    unified_diff: str = ""


@dataclass
class OcrBenchmarkReport:
    """Document-level aggregate OCR benchmark report."""

    avg_cer: float
    avg_wer: float
    avg_character_accuracy: float
    avg_word_accuracy: float

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

    document_cer: float
    document_wer: float
    document_character_accuracy: float
    document_word_accuracy: float

    confidence_high: int
    confidence_medium: int
    confidence_low: int

    lowest_cer_page: int
    highest_cer_page: int
    lowest_cer: float
    highest_cer: float

    pages: list[PageEvaluationResult] = field(default_factory=list)

    ground_truth_metadata: dict[str, str] = field(default_factory=dict)

    total_missing_characters: int = 0
    total_inserted_characters: int = 0
    total_deleted_characters: int = 0
    total_missing_words: int = 0
    total_inserted_words: int = 0
    total_deleted_words: int = 0
    formatting_preservation: float = 1.0
    structural_preservation: float = 1.0
    overall_accuracy: float = 1.0

    processing_time_seconds: float = 0.0

    # Document-level unified diff (debug only)
    unified_diff: str = ""


# ---------------------------------------------------------------------------
# Ground Truth Parsing
# ---------------------------------------------------------------------------


_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:[#>*=\-_\[\](){}|•·–—]+\s*)?"
    r"(?:page|pg|p)\s*[\.:#\-–—]?\s*(?P<num>\d+)\b.*$",
    re.IGNORECASE,
)

_METADATA_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"ground\s*truth(?:\s+transcription)?|"
    r"source(?:\s+file)?|"
    r"purpose|"
    r"validation\s+notes?|"
    r"benchmark\s+(?:notes?|comments?)|"
    r"transcription\s+rules(?:\s+applied)?|"
    r"comments?|"
    r"summary|"
    r"statistics|"
    r"validation\s+statistics|"
    r"ocr\s+statistics|"
    r"document\s+metadata|"
    r"metadata|"
    r"title"
    r")\s*:",
    re.IGNORECASE,
)

_METADATA_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"project\s+\d+|"
    r".*\b(?:verbatim\s+extraction|spelling/grammar|no\s+inference|"
    r"self-corrections?|struck|strikethrough|re-written|rewritten|"
    r"illegible/unreadable|unreadable\s+content|silently\s+dropped|"
    r"ocr-\d+|benchmarking|titus)\b.*"
    r")\s*$",
    re.IGNORECASE,
)

_METADATA_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?"
    r"(?:ground\s*truth(?:\s+transcription)?|source\s+file|purpose|"
    r"validation\s+notes?|benchmark\s+(?:notes?|comments?)|summary|"
    r"statistics|validation\s+statistics|ocr\s+statistics|"
    r"document\s+metadata|metadata)"
    r"\s*$",
    re.IGNORECASE,
)

_APPENDIX_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?"
    r"(?:(?:benchmark\s+(?:notes?|comments?))|summary(?:\s+for\b.*)?|statistics|"
    r"validation\s+statistics|ocr\s+statistics|appendix)"
    r"\s*$",
    re.IGNORECASE,
)

_ANCHOR_RE = re.compile(
    r"^(?:"
    r"section\s+[A-Z0-9IVXLC]+|"
    r"part\s+[A-Z0-9IVXLC]+|"
    r"unit\s+\d+|"
    r"(?:q\.?\s*)?\d+[\.\)]\s+|"
    r"(?:question|ans(?:wer)?)\s*\d*\s*[:\-.]?|"
    r"\([a-zivxlcdm]+\)\s+|"
    r"[a-dA-D][\.\)]\s+|"
    r"[ivxlcdmIVXLCDM]+[\.\)]\s+|"
    r"[-*]\s+|"
    r"instructions?\s*[:\-]"
    r")",
    re.IGNORECASE,
)

_SEPARATOR_RE = re.compile(r"^\s*[-=*_]{3,}\s*$")


class _GroundTruthParseResult(NamedTuple):
    metadata: dict[str, str]
    benchmark_pages: list[str]


def parse_ground_truth(raw_text: str) -> _GroundTruthParseResult:
    """
    Split human ground truth into metadata and benchmark-only page text.

    Metadata and appendices are never returned in benchmark_pages. Users can
    upload rich human files; scoring starts at the first detected OCR-content
    page marker or structural content anchor.
    """
    if not raw_text.strip():
        return _GroundTruthParseResult(metadata={}, benchmark_pages=[])

    lines = _canonical_lines(raw_text)
    benchmark_start_idx = _find_benchmark_start(lines)
    metadata = _extract_metadata(lines[:benchmark_start_idx])
    benchmark_lines = lines[benchmark_start_idx:]

    marker_indices = [
        idx for idx, line in enumerate(benchmark_lines)
        if _is_page_marker(line.strip())
    ]

    if marker_indices:
        pages = _segment_into_pages(benchmark_lines)
    else:
        content = _strip_appendix("\n".join(benchmark_lines))
        pages = [content] if content.strip() else []

    pages = [_strip_appendix(page) for page in pages]
    pages = [page for page in pages if page.strip()]

    return _GroundTruthParseResult(metadata=metadata, benchmark_pages=pages)


def _canonical_lines(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    return text.replace("\r\n", "\n").replace("\r", "\n").splitlines()


def _find_benchmark_start(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if _is_page_marker(line.strip()):
            return idx

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_metadata_line(stripped) or _SEPARATOR_RE.match(stripped):
            continue
        if _ANCHOR_RE.match(stripped):
            return idx

    seen_metadata = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _SEPARATOR_RE.match(stripped):
            continue
        if _is_metadata_line(stripped):
            seen_metadata = True
            continue
        if seen_metadata and _is_metadata_continuation(stripped):
            continue
        if seen_metadata:
            return idx
        return 0

    return len(lines)


def _extract_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    current_heading: str | None = None
    current_values: list[str] = []

    def flush_heading() -> None:
        nonlocal current_heading, current_values
        if current_heading and current_values:
            metadata[current_heading] = " ".join(v.strip() for v in current_values if v.strip())
        current_heading = None
        current_values = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or _SEPARATOR_RE.match(stripped):
            continue

        label_match = _METADATA_LABEL_RE.match(stripped)
        if label_match:
            flush_heading()
            key, value = stripped.split(":", 1)
            key = _clean_metadata_key(key)
            if value.strip():
                metadata[key] = value.strip()
            else:
                current_heading = key
            continue

        if _METADATA_HEADING_RE.match(stripped):
            flush_heading()
            current_heading = _clean_metadata_key(stripped.lstrip("#").strip())
            continue

        if current_heading:
            current_values.append(stripped)

    flush_heading()
    return metadata


def _clean_metadata_key(key: str) -> str:
    key = re.sub(r"^\s*#+\s*", "", key).strip()
    return re.sub(r"\s+", " ", key)


def _is_metadata_line(stripped: str) -> bool:
    return bool(_METADATA_LABEL_RE.match(stripped) or _METADATA_HEADING_RE.match(stripped))


def _is_metadata_continuation(stripped: str) -> bool:
    return bool(_METADATA_CONTINUATION_RE.match(stripped))


def _is_page_marker(stripped: str) -> bool:
    return bool(_PAGE_MARKER_RE.match(stripped))


def _segment_into_pages(lines: list[str]) -> list[str]:
    """Split benchmark lines into page buckets; page marker lines are excluded."""
    pages: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _is_page_marker(stripped):
            if current:
                pages.append(_strip_appendix("\n".join(current).strip()))
            current = []
            continue
        current.append(line)

    if current:
        pages.append(_strip_appendix("\n".join(current).strip()))

    return pages


def _strip_appendix(text: str) -> str:
    """Remove post-benchmark human notes/statistics from a page or document."""
    lines = text.splitlines()
    cut_at = len(lines)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx == 0:
            continue
        if _APPENDIX_HEADING_RE.match(stripped):
            cut_at = idx
            break
        if _METADATA_LABEL_RE.match(stripped):
            remaining = [l.strip() for l in lines[idx:] if l.strip()]
            metadata_like = sum(1 for item in remaining if _METADATA_LABEL_RE.match(item))
            if metadata_like >= max(1, len(remaining) // 2):
                cut_at = idx
                break

    return "\n".join(lines[:cut_at]).strip()


def _strip_trailing_metadata(text: str) -> str:
    """Backward-compatible alias for the old internal helper."""
    return _strip_appendix(text)


# ---------------------------------------------------------------------------
# Text Normalization and Segmentation
# ---------------------------------------------------------------------------


def normalize_text(text: str, punctuation: bool = False) -> str:
    """Normalize text for OCR comparison. Verbatim mode is default."""
    text = unicodedata.normalize("NFC", text)
    text = _strip_scoring_markup(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)

    text = "\n".join(collapsed).strip()

    if punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    return text


def _strip_scoring_markup(text: str) -> str:
    """
    Remove validator-only annotations while preserving the intended text.

    OCR output may contain low-confidence HTML spans, and human ground-truth
    files may contain operator notes such as [LOW CONFIDENCE: best guess "..."].
    These are useful review metadata but not characters to score.
    """
    text = html.unescape(text)
    text = re.sub(
        r"<span\s+[^>]*(?:class=[\"']low-confidence[\"']|data-confidence=[\"']\d+[\"'])[^>]*>(.*?)</span>",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(
        r"\[LOW\s+CONFIDENCE:\s*best\s+guess\s+[\"“](.*?)[\"”][^\]]*\]",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\[LOW\s+CONFIDENCE:[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    filtered_lines: list[str] = []
    skip_continuation = False
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        is_operator_note = (
            lower.startswith("note: word is struck through")
            or "operator self-correction" in lower
            or lower.startswith("cursive rendering is ambiguous")
            or "recommend checking original scan" in lower
            or lower.startswith("marginal note")
            or "not part of verbatim answer" in lower
            or "flagged separately" in lower
        )
        if is_operator_note:
            skip_continuation = True
            continue

        if skip_continuation and (
            "operator self-correction" in lower
            or "recommend checking original scan" in lower
            or "not part of verbatim answer" in lower
            or "flagged separately" in lower
            or stripped.startswith("(")
            or lower.endswith("not spelling.")
            or lower == "directly."
            or lower == "answer body."
        ):
            continue

        skip_continuation = False
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def tokenize_words(text: str) -> list[str]:
    return text.split()


def tokenize_chars(text: str) -> list[str]:
    return list(re.sub(r"\s+", " ", text))


def segment_paragraphs(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"\n\s*\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def segment_lines(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [line for line in normalized.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Edit Distance Computation
# ---------------------------------------------------------------------------


class _EditCounts(NamedTuple):
    reference_len: int
    correct: int
    substituted: int
    deleted: int
    inserted: int


def _compute_edit_counts(reference: list, hypothesis: list) -> _EditCounts:
    """
    Compute edit operation counts using a robust sequence matcher.

    The resulting CER/WER formula is:
      (substitutions + deletions + insertions) / reference length
    """
    if not reference and not hypothesis:
        return _EditCounts(0, 0, 0, 0, 0)
    if not reference:
        return _EditCounts(0, 0, 0, 0, len(hypothesis))
    if not hypothesis:
        return _EditCounts(len(reference), 0, 0, len(reference), 0)

    matcher = difflib.SequenceMatcher(
        isjunk=None,
        a=reference,
        b=hypothesis,
        autojunk=False,
    )

    correct = substituted = deleted = inserted = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_len = i2 - i1
        hyp_len = j2 - j1
        if tag == "equal":
            correct += ref_len
        elif tag == "replace":
            n_sub = min(ref_len, hyp_len)
            substituted += n_sub
            if ref_len > hyp_len:
                deleted += ref_len - hyp_len
            else:
                inserted += hyp_len - ref_len
        elif tag == "delete":
            deleted += ref_len
        elif tag == "insert":
            inserted += hyp_len

    return _EditCounts(
        reference_len=len(reference),
        correct=correct,
        substituted=substituted,
        deleted=deleted,
        inserted=inserted,
    )


def _error_rate(counts: _EditCounts) -> float:
    if counts.reference_len == 0:
        return 0.0 if counts.inserted == 0 else 1.0
    errors = counts.substituted + counts.deleted + counts.inserted
    return min(errors / counts.reference_len, 1.0)


# ---------------------------------------------------------------------------
# Anchor, Formatting, and Structural Metrics
# ---------------------------------------------------------------------------


def detect_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    for line in normalize_text(text).splitlines():
        stripped = line.strip()
        if _ANCHOR_RE.match(stripped):
            anchors.append(stripped)
    return anchors


def _anchor_key(anchor: str) -> str:
    anchor = normalize_text(anchor, punctuation=True).lower()
    return re.sub(r"\s+", " ", anchor).strip()


def _structural_score(gt_text: str, ocr_text: str) -> tuple[float, list[str], list[str]]:
    gt_anchors = detect_anchors(gt_text)
    ocr_keys = {_anchor_key(anchor) for anchor in detect_anchors(ocr_text)}
    if not gt_anchors:
        return 1.0, [], []

    found: list[str] = []
    missing: list[str] = []
    for anchor in gt_anchors:
        if _anchor_key(anchor) in ocr_keys:
            found.append(anchor)
        else:
            missing.append(anchor)

    return len(found) / len(gt_anchors), found, missing


def _formatting_score(gt_text: str, ocr_text: str) -> float:
    gt_lines = segment_lines(gt_text)
    ocr_lines = segment_lines(ocr_text)
    gt_paragraphs = segment_paragraphs(gt_text)
    ocr_paragraphs = segment_paragraphs(ocr_text)

    line_score = _count_similarity(len(gt_lines), len(ocr_lines))
    paragraph_score = _count_similarity(len(gt_paragraphs), len(ocr_paragraphs))
    return (line_score * 0.6) + (paragraph_score * 0.4)


def _count_similarity(reference_count: int, hypothesis_count: int) -> float:
    if reference_count == 0 and hypothesis_count == 0:
        return 1.0
    if reference_count == 0 or hypothesis_count == 0:
        return 0.0
    return max(0.0, 1.0 - abs(reference_count - hypothesis_count) / max(reference_count, hypothesis_count))


def _page_accuracy(char_acc: float, word_acc: float, formatting_score: float, structural_score: float) -> float:
    return (
        char_acc * 0.50
        + word_acc * 0.30
        + formatting_score * 0.10
        + structural_score * 0.10
    )


# ---------------------------------------------------------------------------
# Per-Page Evaluation
# ---------------------------------------------------------------------------


def evaluate_page(page_num: int, ocr_text: str, ground_truth_text: str) -> PageEvaluationResult:
    """Evaluate one OCR page against the matching benchmark ground-truth page."""
    norm_gt = normalize_text(ground_truth_text)
    norm_ocr = normalize_text(ocr_text)

    char_counts = _compute_edit_counts(tokenize_chars(norm_gt), tokenize_chars(norm_ocr))
    cer = _error_rate(char_counts)
    char_acc = 1.0 - cer

    word_counts = _compute_edit_counts(tokenize_words(norm_gt), tokenize_words(norm_ocr))
    wer = _error_rate(word_counts)
    word_acc = 1.0 - wer

    structural_score, anchors_found, anchors_missing = _structural_score(norm_gt, norm_ocr)
    formatting_score = _formatting_score(norm_gt, norm_ocr)
    page_acc = _page_accuracy(char_acc, word_acc, formatting_score, structural_score)

    unified_diff = "\n".join(
        difflib.unified_diff(
            norm_gt.splitlines(),
            norm_ocr.splitlines(),
            fromfile=f"ground_truth_page_{page_num}",
            tofile=f"ocr_output_page_{page_num}",
            lineterm="",
            n=2,
        )
    )

    warnings: list[str] = []
    if char_counts.reference_len == 0:
        warnings.append("Ground truth page is empty.")
    if cer > 0.5:
        warnings.append(f"High CER ({cer:.1%}) - significant OCR errors on this page.")
    if anchors_missing:
        warnings.append(f"Missing {len(anchors_missing)} structural anchor(s).")

    return PageEvaluationResult(
        page=page_num,
        cer=round(cer, 6),
        character_accuracy=round(char_acc, 6),
        chars_reference=char_counts.reference_len,
        chars_correct=char_counts.correct,
        chars_substituted=char_counts.substituted,
        chars_deleted=char_counts.deleted,
        chars_inserted=char_counts.inserted,
        wer=round(wer, 6),
        word_accuracy=round(word_acc, 6),
        words_reference=word_counts.reference_len,
        words_correct=word_counts.correct,
        words_substituted=word_counts.substituted,
        words_deleted=word_counts.deleted,
        words_inserted=word_counts.inserted,
        missing_characters=char_counts.deleted,
        inserted_characters=char_counts.inserted,
        deleted_characters=char_counts.deleted,
        missing_words=word_counts.deleted,
        inserted_words=word_counts.inserted,
        deleted_words=word_counts.deleted,
        formatting_score=round(formatting_score, 6),
        structural_score=round(structural_score, 6),
        page_accuracy=round(page_acc, 6),
        anchors_found=anchors_found[:20],
        anchors_missing=anchors_missing[:20],
        warnings=warnings,
        ground_truth_text=norm_gt,
        ocr_text=norm_ocr,
        unified_diff=unified_diff,
    )


# ---------------------------------------------------------------------------
# Document-Level Evaluation
# ---------------------------------------------------------------------------


def evaluate_benchmark(
    ocr_pages: list[tuple[int, str]],
    ground_truth_raw: str,
) -> OcrBenchmarkReport:
    """
    Main entry point for production-grade OCR benchmarking.

    OCR and ground-truth pages are matched independently by page order. A bad
    page cannot shift or corrupt subsequent page scores.
    """
    started_at = time.perf_counter()

    gt_parse = parse_ground_truth(ground_truth_raw)
    gt_pages = gt_parse.benchmark_pages
    gt_metadata = gt_parse.metadata

    page_results: list[PageEvaluationResult] = []
    for idx, (page_num, ocr_text) in enumerate(ocr_pages):
        gt_text = gt_pages[idx] if idx < len(gt_pages) else ""
        page_results.append(evaluate_page(page_num, ocr_text, gt_text))

    total_char_ref = sum(r.chars_reference for r in page_results)
    total_char_correct = sum(r.chars_correct for r in page_results)
    total_char_sub = sum(r.chars_substituted for r in page_results)
    total_char_del = sum(r.chars_deleted for r in page_results)
    total_char_ins = sum(r.chars_inserted for r in page_results)

    total_word_ref = sum(r.words_reference for r in page_results)
    total_word_correct = sum(r.words_correct for r in page_results)
    total_word_sub = sum(r.words_substituted for r in page_results)
    total_word_del = sum(r.words_deleted for r in page_results)
    total_word_ins = sum(r.words_inserted for r in page_results)

    doc_char_counts = _EditCounts(total_char_ref, total_char_correct, total_char_sub, total_char_del, total_char_ins)
    doc_word_counts = _EditCounts(total_word_ref, total_word_correct, total_word_sub, total_word_del, total_word_ins)
    doc_cer = _error_rate(doc_char_counts)
    doc_wer = _error_rate(doc_word_counts)

    if page_results:
        avg_cer = sum(r.cer for r in page_results) / len(page_results)
        avg_wer = sum(r.wer for r in page_results) / len(page_results)
        avg_char_acc = sum(r.character_accuracy for r in page_results) / len(page_results)
        avg_word_acc = sum(r.word_accuracy for r in page_results) / len(page_results)
        formatting_preservation = sum(r.formatting_score for r in page_results) / len(page_results)
        structural_preservation = sum(r.structural_score for r in page_results) / len(page_results)
        overall_accuracy = sum(r.page_accuracy for r in page_results) / len(page_results)
    else:
        avg_cer = avg_wer = 0.0
        avg_char_acc = avg_word_acc = 1.0
        formatting_preservation = structural_preservation = overall_accuracy = 1.0

    conf_high = sum(1 for r in page_results if r.character_accuracy >= 0.95)
    conf_medium = sum(1 for r in page_results if 0.80 <= r.character_accuracy < 0.95)
    conf_low = sum(1 for r in page_results if r.character_accuracy < 0.80)

    if page_results:
        best = min(page_results, key=lambda r: r.cer)
        worst = max(page_results, key=lambda r: r.cer)
        lowest_cer_page = best.page
        highest_cer_page = worst.page
        lowest_cer = best.cer
        highest_cer = worst.cer
    else:
        lowest_cer_page = highest_cer_page = 0
        lowest_cer = highest_cer = 0.0

    combined_gt = "\n\n".join(gt_pages)
    combined_ocr = "\n\n".join(text for _, text in ocr_pages)
    norm_cgt = normalize_text(combined_gt)
    norm_cocr = normalize_text(combined_ocr)
    doc_unified_diff = "\n".join(
        difflib.unified_diff(
            norm_cgt.splitlines(),
            norm_cocr.splitlines(),
            fromfile="ground_truth",
            tofile="ocr_output",
            lineterm="",
            n=2,
        )
    )

    elapsed = time.perf_counter() - started_at

    return OcrBenchmarkReport(
        avg_cer=round(avg_cer, 6),
        avg_wer=round(avg_wer, 6),
        avg_character_accuracy=round(avg_char_acc, 6),
        avg_word_accuracy=round(avg_word_acc, 6),
        total_chars_reference=total_char_ref,
        total_chars_correct=total_char_correct,
        total_chars_substituted=total_char_sub,
        total_chars_deleted=total_char_del,
        total_chars_inserted=total_char_ins,
        total_words_reference=total_word_ref,
        total_words_correct=total_word_correct,
        total_words_substituted=total_word_sub,
        total_words_deleted=total_word_del,
        total_words_inserted=total_word_ins,
        document_cer=round(doc_cer, 6),
        document_wer=round(doc_wer, 6),
        document_character_accuracy=round(1.0 - doc_cer, 6),
        document_word_accuracy=round(1.0 - doc_wer, 6),
        confidence_high=conf_high,
        confidence_medium=conf_medium,
        confidence_low=conf_low,
        lowest_cer_page=lowest_cer_page,
        highest_cer_page=highest_cer_page,
        lowest_cer=round(lowest_cer, 6),
        highest_cer=round(highest_cer, 6),
        pages=page_results,
        ground_truth_metadata=gt_metadata,
        total_missing_characters=total_char_del,
        total_inserted_characters=total_char_ins,
        total_deleted_characters=total_char_del,
        total_missing_words=total_word_del,
        total_inserted_words=total_word_ins,
        total_deleted_words=total_word_del,
        formatting_preservation=round(formatting_preservation, 6),
        structural_preservation=round(structural_preservation, 6),
        overall_accuracy=round(overall_accuracy, 6),
        processing_time_seconds=round(elapsed, 6),
        unified_diff=doc_unified_diff,
    )
