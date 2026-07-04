import argparse
import difflib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationReport:
    character_accuracy: float
    word_accuracy: float
    missing_characters: int
    extra_characters: int
    processing_time_seconds: float
    unified_diff: str


def evaluate_text(ocr_text: str, ground_truth_text: str) -> EvaluationReport:
    started_at = time.perf_counter()
    normalized_ocr = _normalize_newlines(ocr_text)
    normalized_truth = _normalize_newlines(ground_truth_text)

    character_accuracy = _sequence_accuracy(normalized_ocr, normalized_truth)
    word_accuracy = _sequence_accuracy(
        normalized_ocr.split(),
        normalized_truth.split(),
    )
    missing_characters, extra_characters = _character_delta_counts(
        normalized_ocr,
        normalized_truth,
    )
    unified_diff = "\n".join(
        difflib.unified_diff(
            normalized_truth.splitlines(),
            normalized_ocr.splitlines(),
            fromfile="ground_truth",
            tofile="ocr_output",
            lineterm="",
        )
    )

    return EvaluationReport(
        character_accuracy=round(character_accuracy, 6),
        word_accuracy=round(word_accuracy, 6),
        missing_characters=missing_characters,
        extra_characters=extra_characters,
        processing_time_seconds=round(time.perf_counter() - started_at, 6),
        unified_diff=unified_diff,
    )


def evaluate_files(ocr_output_path: Path, ground_truth_path: Path) -> EvaluationReport:
    return evaluate_text(
        ocr_output_path.read_text(encoding="utf-8"),
        ground_truth_path.read_text(encoding="utf-8"),
    )


def _sequence_accuracy(actual: str | list[str], expected: str | list[str]) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    return matches / max(len(expected), len(actual))


def _character_delta_counts(actual: str, expected: str) -> tuple[int, int]:
    missing = 0
    extra = 0
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)

    for tag, expected_start, expected_end, actual_start, actual_end in matcher.get_opcodes():
        expected_len = expected_end - expected_start
        actual_len = actual_end - actual_start
        if tag == "delete":
            missing += expected_len
        elif tag == "insert":
            extra += actual_len
        elif tag == "replace":
            missing += expected_len
            extra += actual_len

    return missing, extra


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR text against ground truth for Project TITUS-082.",
    )
    parser.add_argument("ocr_output", type=Path, help="Path to OCR output text.")
    parser.add_argument("ground_truth", type=Path, help="Path to ground-truth text.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    args = parser.parse_args()

    report = evaluate_files(args.ocr_output, args.ground_truth)
    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return

    print(f"Character accuracy: {report.character_accuracy:.2%}")
    print(f"Word accuracy: {report.word_accuracy:.2%}")
    print(f"Missing characters: {report.missing_characters}")
    print(f"Extra characters: {report.extra_characters}")
    print(f"Processing time: {report.processing_time_seconds:.6f}s")
    print("Unified diff:")
    print(report.unified_diff or "(no differences)")


if __name__ == "__main__":
    main()
