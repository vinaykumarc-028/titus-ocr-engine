import unittest

from evaluate import evaluate_text
from evaluate_ocr import (
    evaluate_benchmark,
    evaluate_page,
    parse_ground_truth,
    normalize_text,
    detect_anchors,
    _compute_edit_counts,
    _error_rate,
)
from app.ocr.parser import parse_markdown


class EvaluateTests(unittest.TestCase):
    """Existing legacy tests -- must remain passing for backward compat."""

    def test_exact_match_scores_full_accuracy(self) -> None:
        report = evaluate_text("A quick test.", "A quick test.")

        self.assertEqual(report.character_accuracy, 1.0)
        self.assertEqual(report.word_accuracy, 1.0)
        self.assertEqual(report.missing_characters, 0)
        self.assertEqual(report.extra_characters, 0)
        self.assertEqual(report.unified_diff, "")

    def test_difference_report_includes_diff(self) -> None:
        report = evaluate_text("A quik test!", "A quick test.")

        self.assertLess(report.character_accuracy, 1.0)
        self.assertLess(report.word_accuracy, 1.0)
        self.assertGreater(report.missing_characters, 0)
        self.assertGreater(report.extra_characters, 0)
        self.assertIn("--- ground_truth", report.unified_diff)
        self.assertIn("+++ ocr_output", report.unified_diff)


class OcrBenchmarkTests(unittest.TestCase):
    """Production-grade OCR benchmarking engine tests."""

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def test_normalize_collapses_whitespace(self) -> None:
        result = normalize_text("Hello   World\r\n")
        self.assertEqual(result, "Hello World")

    def test_normalize_unicode_nfc(self) -> None:
        decomposed = "e\u0301"
        precomposed = "\u00e9"
        self.assertEqual(normalize_text(decomposed), normalize_text(precomposed))

    def test_low_confidence_html_span_not_scored_as_text(self) -> None:
        text = 'Ans 9 <span class="low-confidence" data-confidence="65">waited</span>'
        self.assertEqual(normalize_text(text), "Ans 9 waited")

    def test_parser_plain_text_strips_low_confidence_span(self) -> None:
        page = parse_markdown(
            1,
            'Ans 9 <span class="low-confidence" data-confidence="65">waited</span>',
        )
        self.assertEqual(page.plain_text, "Ans 9 waited")

    # ------------------------------------------------------------------
    # Ground Truth Parsing -- Metadata Stripping
    # ------------------------------------------------------------------

    def test_metadata_stripped_before_benchmarking(self) -> None:
        gt = (
            "Ground Truth: My Exam Doc\n"
            "Source File: exam.pdf\n"
            "Purpose: Benchmarking\n"
            "Validation Notes: None\n"
            "Summary: First test\n"
            "\n"
            "Section A\n"
            "1. What is photosynthesis?\n"
        )
        result = parse_ground_truth(gt)
        self.assertIn("Ground Truth", result.metadata)
        full_benchmark = "\n".join(result.benchmark_pages)
        self.assertNotIn("Source File", full_benchmark)
        self.assertNotIn("Purpose", full_benchmark)
        self.assertIn("Section A", full_benchmark)

    def test_empty_ground_truth_produces_empty_pages(self) -> None:
        result = parse_ground_truth("")
        self.assertEqual(result.benchmark_pages, [])

    def test_ground_truth_without_metadata_passes_through(self) -> None:
        gt = "Section A\n1. What is water?\n2. Define gravity."
        result = parse_ground_truth(gt)
        self.assertEqual(len(result.benchmark_pages), 1)
        self.assertIn("Section A", result.benchmark_pages[0])

    def test_page_markers_split_into_multiple_pages(self) -> None:
        gt = (
            "--- Page 1 ---\n"
            "Section A\n"
            "1. Question one\n"
            "--- Page 2 ---\n"
            "Section B\n"
            "2. Question two\n"
        )
        result = parse_ground_truth(gt)
        self.assertEqual(len(result.benchmark_pages), 2)
        self.assertIn("Section A", result.benchmark_pages[0])
        self.assertIn("Section B", result.benchmark_pages[1])

    # ------------------------------------------------------------------
    # Edit Distance / CER / WER
    # ------------------------------------------------------------------

    def test_perfect_match_zero_errors(self) -> None:
        tokens = list("hello world")
        counts = _compute_edit_counts(tokens, tokens)
        self.assertEqual(_error_rate(counts), 0.0)
        self.assertEqual(counts.correct, len(tokens))
        self.assertEqual(counts.substituted, 0)
        self.assertEqual(counts.deleted, 0)
        self.assertEqual(counts.inserted, 0)

    def test_complete_mismatch_cer_is_one(self) -> None:
        ref = list("abc")
        hyp = list("xyz")
        counts = _compute_edit_counts(ref, hyp)
        self.assertEqual(_error_rate(counts), 1.0)

    def test_empty_reference_empty_hypothesis_zero_error(self) -> None:
        counts = _compute_edit_counts([], [])
        self.assertEqual(_error_rate(counts), 0.0)

    def test_empty_hypothesis_cer_is_one(self) -> None:
        counts = _compute_edit_counts(list("hello"), [])
        self.assertEqual(_error_rate(counts), 1.0)

    # ------------------------------------------------------------------
    # Per-Page Evaluation
    # ------------------------------------------------------------------

    def test_exact_page_match(self) -> None:
        text = "Section A\n1. What is photosynthesis?\n(a) Define chlorophyll."
        result = evaluate_page(1, text, text)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.character_accuracy, 1.0)
        self.assertEqual(result.word_accuracy, 1.0)

    def test_one_word_typo_small_cer(self) -> None:
        gt = "The quick brown fox"
        ocr = "The quik brown fox"
        result = evaluate_page(1, ocr, gt)
        self.assertGreater(result.cer, 0.0)
        self.assertLess(result.cer, 0.2)
        self.assertGreater(result.character_accuracy, 0.8)

    def test_single_page_error_does_not_cascade(self) -> None:
        """Core requirement: one page with errors must not cascade to other pages."""
        gt_page1 = "AAAAAAAAAAAAAAAAAAA"
        gt_page2 = "BBBBBBBBBBBBBBBBBBB"
        ocr_page1 = "XXXXXXXXXXXXXXXXXXX"
        ocr_page2 = "BBBBBBBBBBBBBBBBBBB"

        report = evaluate_benchmark(
            [(1, ocr_page1), (2, ocr_page2)],
            f"--- Page 1 ---\n{gt_page1}\n--- Page 2 ---\n{gt_page2}"
        )

        page1_result = next(p for p in report.pages if p.page == 1)
        page2_result = next(p for p in report.pages if p.page == 2)

        self.assertEqual(page2_result.cer, 0.0)
        self.assertEqual(page2_result.character_accuracy, 1.0)
        self.assertGreater(page1_result.cer, 0.5)

    # ------------------------------------------------------------------
    # Document-Level Evaluation
    # ------------------------------------------------------------------

    def test_metadata_not_in_score(self) -> None:
        """Metadata preamble must never affect accuracy."""
        gt_with_metadata = (
            "Ground Truth: Exam Paper\n"
            "Source: english1.pdf\n"
            "Purpose: Validation\n"
            "Summary: 3-page exam\n"
            "\n"
            "--- Page 1 ---\n"
            "Hello world\n"
        )
        gt_clean = "--- Page 1 ---\nHello world\n"

        ocr = [(1, "Hello world")]

        report_with_meta = evaluate_benchmark(ocr, gt_with_metadata)
        report_clean = evaluate_benchmark(ocr, gt_clean)

        self.assertEqual(report_with_meta.document_cer, report_clean.document_cer)
        self.assertEqual(report_with_meta.document_character_accuracy, 1.0)
        self.assertEqual(report_clean.document_character_accuracy, 1.0)

    def test_human_ground_truth_header_never_scores_without_page_markers(self) -> None:
        gt = (
            "Ground Truth Transcription\n"
            "Source File: math-paper.pdf\n"
            "Purpose: OCR validation\n"
            "Validation Notes: Compare only transcription content.\n"
            "Summary: Human notes for reviewer.\n"
            "\n"
            "Section A\n"
            "1. What is 2 + 2?\n"
        )

        report = evaluate_benchmark(
            [(1, "Section A\n1. What is 2 + 2?")],
            gt,
        )

        self.assertEqual(report.document_character_accuracy, 1.0)
        self.assertIn("Source File", report.ground_truth_metadata)
        self.assertNotIn("Source File", report.pages[0].ground_truth_text)
        self.assertNotIn("Validation Notes", report.pages[0].ground_truth_text)

    def test_titus_ground_truth_preamble_never_enters_benchmark_diff(self) -> None:
        gt = (
            "Ground Truth Transcription\n"
            "Source file: english_1.pdf\n"
            "Purpose: Manual verbatim ground truth for OCR validation "
            "(Mistral OCR 3 / Pixtral 12B benchmarking) - TITUS\n"
            "Project 082\n"
            "Transcription rules applied: Verbatim extraction only. "
            "No spelling/grammar correction. No inference.\n"
            "self-corrections preserved as written (original struck replacement, "
            "shown here as struck-through text followed by\n"
            "the re-written word). Illegible/unreadable content flagged, never "
            "silently dropped, per OCR-02 / OCR-03 / OCR-18.\n"
            "Page 1 (sheet no. \"2\")\n"
            "[BLANK - only faint bleed-through/ghosting from the reverse page is visible.]\n"
            "Page 2 (sheet no. \"3\")\n"
            "English\n"
            "Language and Literature\n"
        )

        report = evaluate_benchmark(
            [
                (1, "[BLANK - only faint bleed-through/ghosting from the reverse page is visible.]"),
                (2, "English\nLanguage and Literature"),
            ],
            gt,
        )

        self.assertEqual(report.document_character_accuracy, 1.0)
        self.assertNotIn("Ground Truth Transcription", report.unified_diff)
        self.assertNotIn("Source file", report.unified_diff)
        self.assertNotIn("Transcription rules applied", report.unified_diff)
        self.assertNotIn("Project 082", report.pages[0].ground_truth_text)

    def test_titus_metadata_continuations_are_skipped_until_content_anchor(self) -> None:
        gt = (
            "Ground Truth Transcription\n"
            "Source file: english_1.pdf\n"
            "Purpose: Manual verbatim ground truth for OCR validation - TITUS\n"
            "Project 082\n"
            "Transcription rules applied: Verbatim extraction only. No inference.\n"
            "self-corrections preserved as written\n"
            "the re-written word). Illegible/unreadable content flagged, per OCR-02.\n"
            "Section A\n"
            "1. Read the following passage.\n"
        )

        report = evaluate_benchmark(
            [(1, "Section A\n1. Read the following passage.")],
            gt,
        )

        self.assertEqual(report.document_character_accuracy, 1.0)
        self.assertNotIn("Ground Truth Transcription", report.pages[0].ground_truth_text)
        self.assertNotIn("Transcription rules applied", report.pages[0].ground_truth_text)

    def test_ground_truth_operator_annotations_do_not_reduce_accuracy(self) -> None:
        gt = (
            "--- Page 1 ---\n"
            "(x)\n"
            "(D) [LOW CONFIDENCE: best guess \"waited\"]\n"
            "Cursive rendering is ambiguous between \"waited\" and a similarly-shaped word; recommend checking original scan\n"
            "directly.\n"
            "he had been very occupied occupied with work that week\n"
            "Note: word is struck through and rewritten identically as \"occupied\" in the original - preserved as written; possible\n"
            "operator self-correction of handwriting, not spelling.\n"
        )

        ocr = (
            "(x)\n"
            "(D) waited\n"
            "he had been very occupied occupied with work that week\n"
        )

        report = evaluate_benchmark([(1, ocr)], gt)

        self.assertEqual(report.document_character_accuracy, 1.0)
        self.assertNotIn("LOW CONFIDENCE", report.pages[0].ground_truth_text)
        self.assertNotIn("Cursive rendering", report.pages[0].ground_truth_text)
        self.assertNotIn("operator self-correction", report.pages[0].ground_truth_text)

    def test_ground_truth_marginal_note_annotation_does_not_score(self) -> None:
        gt = (
            "--- Page 1 ---\n"
            "The given data shows an upward trend in gym membership.\n"
            "Marginal note at bottom left of page, unrelated to main answer - appears to be a rough numeric calculation\n"
            "(\"Eighty only\", fraction-like figures). Not part of verbatim answer text; flagged separately rather than merged into the\n"
            "answer body.\n"
        )

        report = evaluate_benchmark(
            [(1, "The given data shows an upward trend in gym membership.")],
            gt,
        )

        self.assertEqual(report.document_character_accuracy, 1.0)
        self.assertNotIn("Marginal note", report.pages[0].ground_truth_text)
        self.assertNotIn("Eighty only", report.pages[0].ground_truth_text)

    def test_appendix_statistics_never_score(self) -> None:
        gt = (
            "--- Page 1 ---\n"
            "Section A\n"
            "1. Define photosynthesis.\n"
            "\n"
            "Validation Statistics\n"
            "Character Count: 1000\n"
            "Benchmark Comments: Human-only note\n"
        )

        report = evaluate_benchmark(
            [(1, "Section A\n1. Define photosynthesis.")],
            gt,
        )

        self.assertEqual(report.document_character_accuracy, 1.0)
        self.assertNotIn("Validation Statistics", report.pages[0].ground_truth_text)
        self.assertNotIn("Benchmark Comments", report.pages[0].ground_truth_text)

    def test_missing_heading_recovers_alignment_for_remaining_text(self) -> None:
        gt = "--- Page 1 ---\nSection A\n1. Alpha\n2. Beta\n3. Gamma\n"
        ocr = "1. Alpha\n2. Beta\n3. Gamma\n"

        report = evaluate_benchmark([(1, ocr)], gt)

        self.assertGreater(report.document_character_accuracy, 0.70)
        self.assertLess(report.document_cer, 0.30)
        self.assertEqual(report.total_chars_inserted, 0)

    def test_confidence_distribution_counts(self) -> None:
        # Page 1: exact match -> char_acc=1.0 -> high (>=0.95)
        # Page 2: 2 chars wrong out of 21 -> CER=2/21~0.095 -> acc~0.905 -> medium [0.80, 0.95)
        # Page 3: all wrong -> char_acc~0.0 -> low (<0.80)
        gt_p1 = "ABCDEFGHIJ"
        gt_p2 = "ABCDEFGHIJKLMNOPQRSTU"   # 21 chars
        gt_p3 = "ABCDEFGHIJ"
        gt = (
            f"--- Page 1 ---\n{gt_p1}\n"
            f"--- Page 2 ---\n{gt_p2}\n"
            f"--- Page 3 ---\n{gt_p3}\n"
        )
        ocr = [
            (1, gt_p1),                              # exact match
            (2, "ABCXEFGHIJKLMNOPQRXTU"),            # 2 chars wrong -> acc~0.905 -> medium
            (3, "ZZZZZZZZZZ"),                        # all wrong -> low
        ]
        report = evaluate_benchmark(ocr, gt)
        self.assertEqual(report.confidence_high, 1)
        self.assertEqual(report.confidence_medium, 1)
        self.assertEqual(report.confidence_low, 1)

    def test_single_block_gt_vs_multi_page_ocr(self) -> None:
        """GT without page markers: single block compared to each OCR page."""
        gt = "The quick brown fox"
        ocr = [(1, "The quick brown fox"), (2, "jumped over")]
        report = evaluate_benchmark(ocr, gt)
        page1 = next(p for p in report.pages if p.page == 1)
        self.assertEqual(page1.cer, 0.0)

    # ------------------------------------------------------------------
    # Anchor Detection
    # ------------------------------------------------------------------

    def test_anchor_detection_finds_sections(self) -> None:
        text = "Section A\nQuestion 1. What is water?\nSection B\n"
        anchors = detect_anchors(text)
        self.assertTrue(any("Section A" in a for a in anchors))

    def test_anchor_detection_empty_text(self) -> None:
        self.assertEqual(detect_anchors(""), [])


if __name__ == "__main__":
    unittest.main()
