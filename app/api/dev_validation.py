import base64
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, StreamingResponse

from pydantic import BaseModel
from app.config import settings
from app.core.image import (
    ImageProcessingError,
    preprocess_image,
    preprocess_image_with_metrics,
    normalize_page_image_with_metrics,
)
from app.core.pdf import PDFProcessingError, split_pdf_to_images
from app.core.tempfiles import LocalTempStorage
from app.core.validator import FileValidationError, UploadedFile, validate_upload
from app.ocr.intelligence import analyze_examination_page
from app.ocr.models import (
    EvaluationResult,
    OcrBenchmarkReport,
    OCRValidationPage,
    OCRValidationResponse,
    PageMetrics,
    OCRRunSummary,
    DocumentElement,
)
from app.ocr.service import OCRService, OCRServiceError, OCRTimeoutError
from evaluate import evaluate_text
from evaluate_ocr import evaluate_benchmark

logger = logging.getLogger("ocr_pipeline")
router = APIRouter(prefix="/dev/ocr-validation", tags=["dev-ocr-validation"])


@router.get("", response_class=HTMLResponse)
async def validation_interface() -> HTMLResponse:
    return HTMLResponse(_VALIDATION_HTML)


@router.post("/process")
async def process_validation_upload(
    files: list[UploadFile] = File(...),
    ground_truth: UploadFile | None = File(default=None),
) -> StreamingResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one handwritten PDF or image is required.",
        )

    if ground_truth is not None:
        gt_filename = (ground_truth.filename or "").lower()
        if not (gt_filename.endswith(".txt") or gt_filename.endswith(".pdf")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ground truth must be a .txt or .pdf file.",
            )

    async def event_generator():
        storage = LocalTempStorage(settings.temp_root)
        total_started_at = time.perf_counter()
        pages: list[OCRValidationPage] = []
        warnings_list: list[str] = []
        failures_list: list[str] = []
        ocr_service: OCRService | None = None

        try:
            # Try loading ground truth text
            ground_truth_text: str | None = None
            if ground_truth is not None:
                gt_filename = (ground_truth.filename or "").lower()
                if gt_filename.endswith(".pdf"):
                    import pypdfium2 as pdfium
                    pdf_data = await ground_truth.read()
                    try:
                        document = pdfium.PdfDocument(pdf_data)
                        text_list = []
                        for page in document:
                            textpage = page.get_textpage()
                            text_list.append(textpage.get_text_bounded())
                            page.close()
                        document.close()
                        ground_truth_text = "\n".join(text_list)
                    except Exception as e:
                        logger.error(f"Failed to extract text from ground truth PDF: {e}")
                        yield f"data: {json.dumps({'type': 'error', 'category': 'validation_error', 'message': f'Failed to parse ground truth PDF: {str(e)}'})}\n\n"
                        return
                else:
                    ground_truth_text = (await ground_truth.read()).decode("utf-8-sig")

            yield f"data: {json.dumps({'type': 'status', 'message': 'Uploading files...'})}\n\n"

            async with storage:
                # 1. Save uploads and split PDF or preprocess image
                yield f"data: {json.dumps({'type': 'status', 'message': 'Splitting PDF/images...'})}\n\n"

                page_tasks = []
                current_page_num = 1

                for upload in files:
                    content = await upload.read()
                    uploaded_file = UploadedFile(
                        filename=upload.filename or "upload",
                        content_type=upload.content_type,
                        size=len(content),
                        data=content,
                    )

                    try:
                        validated = validate_upload(uploaded_file)
                    except FileValidationError as exc:
                        yield f"data: {json.dumps({'type': 'error', 'category': 'validation_error', 'message': f'Validation failed for {upload.filename}: {str(exc)}'})}\n\n"
                        return

                    stored_upload = await storage.save_upload(
                        validated.filename,
                        validated.data,
                    )

                    if validated.kind == "pdf":
                        try:
                            page_images = await split_pdf_to_images(
                                stored_upload,
                                storage,
                                start_page=current_page_num,
                                display_name=validated.filename,
                            )
                            for page_img in page_images:
                                page_tasks.append((current_page_num, page_img, validated.filename, True))
                                current_page_num += 1
                        except PDFProcessingError as exc:
                            yield f"data: {json.dumps({'type': 'error', 'category': 'invalid_pdf', 'message': f'PDF processing failed for {validated.filename}: {str(exc)}'})}\n\n"
                            return
                        except Exception as exc:
                            yield f"data: {json.dumps({'type': 'error', 'category': 'invalid_pdf', 'message': f'PDF processing failed: {str(exc)}'})}\n\n"
                            return
                    else:
                        try:
                            # Preprocess single image upload
                            page_img = await preprocess_image(
                                stored_upload,
                                storage,
                                page_number=current_page_num,
                                display_name=validated.filename,
                            )
                            page_tasks.append((current_page_num, page_img, validated.filename, False))
                            current_page_num += 1
                        except ImageProcessingError as exc:
                            yield f"data: {json.dumps({'type': 'error', 'category': 'corrupted_image', 'message': f'Image processing failed for {validated.filename}: {str(exc)}'})}\n\n"
                            return
                        except Exception as exc:
                            yield f"data: {json.dumps({'type': 'error', 'category': 'corrupted_image', 'message': f'Image processing failed: {str(exc)}'})}\n\n"
                            return

                total_pages = len(page_tasks)
                if total_pages > settings.max_pages_per_job:
                    yield f"data: {json.dumps({'type': 'error', 'category': 'validation_error', 'message': f'A single validation run can contain up to {settings.max_pages_per_job} pages.'})}\n\n"
                    return

                # Yield init_pages event
                yield f"data: {json.dumps({'type': 'init_pages', 'total_pages': total_pages})}\n\n"

                # Initialize OCR Service
                try:
                    ocr_service = OCRService()
                except RuntimeError as exc:
                    yield f"data: {json.dumps({'type': 'error', 'category': 'authentication_failure', 'message': str(exc)})}\n\n"
                    return

                # 2. Run OCR on each page
                for page_num, page_image_path, source_filename, is_pdf in page_tasks:
                    yield f"data: {json.dumps({'type': 'page_status', 'page': page_num, 'status': 'running'})}\n\n"

                    page_started_at = time.perf_counter()
                    preprocessing_duration = 0.0
                    image_size = 0
                    image_resolution = ""
                    gemini_duration = 0.0
                    parsing_duration = 0.0
                    retry_count = 0

                    try:
                        # Preprocessing / Normalize
                        if is_pdf:
                            preprocess_res = await normalize_page_image_with_metrics(
                                page_image_path, display_name=source_filename
                            )
                        else:
                            preprocess_res = await normalize_page_image_with_metrics(
                                page_image_path, display_name=source_filename
                            )

                        preprocessing_duration = preprocess_res.duration_seconds
                        image_size = preprocess_res.metadata.size_bytes
                        image_resolution = f"{preprocess_res.metadata.width}x{preprocess_res.metadata.height}"
                        warnings_list.extend(preprocess_res.warnings)

                        # Gemini OCR
                        ocr_res = await ocr_service.extract_page(page_num, preprocess_res.path)
                        gemini_duration = ocr_res.gemini_request_seconds
                        retry_count = ocr_res.retry_count

                        # Parse Markdown
                        parsing_start = time.perf_counter()
                        parsed_page = analyze_examination_page(page_num, ocr_res.markdown)
                        parsing_duration = time.perf_counter() - parsing_start

                        total_page_time = time.perf_counter() - page_started_at

                        page_result = OCRValidationPage(
                            page=page_num,
                            status="Completed",
                            image_data_url=_image_data_url(preprocess_res.path),
                            markdown=parsed_page.markdown,
                            plain_text=parsed_page.plain_text,
                            elements=parsed_page.elements,
                            processing_time_seconds=round(total_page_time, 6),
                            metrics=PageMetrics(
                                preprocessing_time_seconds=round(preprocessing_duration, 6),
                                image_size_bytes=image_size,
                                image_resolution=image_resolution,
                                gemini_request_seconds=round(gemini_duration, 6),
                                parsing_time_seconds=round(parsing_duration, 6),
                                total_time_seconds=round(total_page_time, 6),
                            ),
                            retry_count=retry_count,
                            quality_report=preprocess_res.quality_report,
                        )
                        pages.append(page_result)

                        logger.info(
                            f"Page processed: filename={source_filename}, page_number={page_num}, "
                            f"image_size={image_size}, resolution={image_resolution}, "
                            f"preprocessing_duration={preprocessing_duration:.4f}s, "
                            f"gemini_request_duration={gemini_duration:.4f}s, "
                            f"parser_duration={parsing_duration:.4f}s, "
                            f"total_duration={total_page_time:.4f}s, "
                            f"retry_count={retry_count}, final_status=Completed"
                        )

                        yield f"data: {json.dumps({'type': 'page_status', 'page': page_num, 'status': 'completed', 'duration': total_page_time})}\n\n"

                    except Exception as exc:
                        total_page_time = time.perf_counter() - page_started_at

                        category = "unknown_exception"
                        if isinstance(exc, ImageProcessingError):
                            category = "corrupted_image"
                        elif isinstance(exc, OCRTimeoutError):
                            category = "gemini_api_timeout"
                        elif isinstance(exc, OCRServiceError):
                            category = exc.category

                        if 'ocr_res' in locals() and 'parsed_page' not in locals():
                            category = "parsing_failure"

                        reason_msg = "OCR exceeded timeout" if category == "gemini_api_timeout" else str(exc)
                        error_reason = reason_msg
                        failures_list.append(f"Page {page_num}: {category} - {error_reason}")

                        logger.error(
                            f"Page processed with error: filename={source_filename}, page_number={page_num}, "
                            f"image_size={image_size}, resolution={image_resolution}, "
                            f"preprocessing_duration={preprocessing_duration:.4f}s, "
                            f"gemini_request_duration={gemini_duration:.4f}s, "
                            f"parser_duration={parsing_duration:.4f}s, "
                            f"total_duration={total_page_time:.4f}s, "
                            f"retry_count={retry_count}, final_status={category}, error={error_reason}"
                        )

                        img_url = ""
                        try:
                            img_url = _image_data_url(page_image_path)
                        except Exception:
                            pass

                        status_str = "Timeout" if category == "gemini_api_timeout" else "Failed"

                        page_result = OCRValidationPage(
                            page=page_num,
                            status=status_str,
                            error_reason=error_reason,
                            image_data_url=img_url,
                            processing_time_seconds=round(total_page_time, 6),
                            metrics=PageMetrics(
                                preprocessing_time_seconds=round(preprocessing_duration, 6),
                                image_size_bytes=image_size,
                                image_resolution=image_resolution,
                                gemini_request_seconds=round(gemini_duration, 6),
                                parsing_time_seconds=round(parsing_duration, 6),
                                total_time_seconds=round(total_page_time, 6),
                            ),
                            retry_count=retry_count,
                            quality_report=preprocess_res.quality_report if 'preprocess_res' in locals() else None,
                        )
                        pages.append(page_result)

                        yield f"data: {json.dumps({'type': 'page_status', 'page': page_num, 'status': status_str.lower(), 'duration': total_page_time, 'reason': error_reason})}\n\n"

                # 3. Evaluation
                evaluation = None
                benchmark = None
                successful_pages = [p for p in pages if p.status == "Completed"]

                if ground_truth_text is not None and successful_pages:
                    # Production-grade benchmark: pass individual page texts so
                    # page-level alignment can be performed. Metadata in the ground
                    # truth is automatically stripped before scoring.
                    ocr_page_tuples = [
                        (p.page, p.plain_text) for p in successful_pages
                    ]
                    try:
                        from app.ocr.models import PageEvaluationResult as PydanticPageResult
                        benchmark_result = evaluate_benchmark(ocr_page_tuples, ground_truth_text)

                        # Convert page results from evaluate_ocr dataclasses to Pydantic models
                        pydantic_pages = [
                            PydanticPageResult(
                                page=pr.page,
                                cer=pr.cer,
                                character_accuracy=pr.character_accuracy,
                                chars_reference=pr.chars_reference,
                                chars_correct=pr.chars_correct,
                                chars_substituted=pr.chars_substituted,
                                chars_deleted=pr.chars_deleted,
                                chars_inserted=pr.chars_inserted,
                                wer=pr.wer,
                                word_accuracy=pr.word_accuracy,
                                words_reference=pr.words_reference,
                                words_correct=pr.words_correct,
                                words_substituted=pr.words_substituted,
                                words_deleted=pr.words_deleted,
                                words_inserted=pr.words_inserted,
                                missing_characters=pr.missing_characters,
                                inserted_characters=pr.inserted_characters,
                                deleted_characters=pr.deleted_characters,
                                missing_words=pr.missing_words,
                                inserted_words=pr.inserted_words,
                                deleted_words=pr.deleted_words,
                                formatting_score=pr.formatting_score,
                                structural_score=pr.structural_score,
                                page_accuracy=pr.page_accuracy,
                                anchors_found=pr.anchors_found,
                                anchors_missing=pr.anchors_missing,
                                warnings=pr.warnings,
                                ground_truth_text=pr.ground_truth_text,
                                ocr_text=pr.ocr_text,
                                unified_diff=pr.unified_diff,
                            )
                            for pr in benchmark_result.pages
                        ]

                        # Convert document-level result to Pydantic OcrBenchmarkReport
                        benchmark = OcrBenchmarkReport(
                            avg_cer=benchmark_result.avg_cer,
                            avg_wer=benchmark_result.avg_wer,
                            avg_character_accuracy=benchmark_result.avg_character_accuracy,
                            avg_word_accuracy=benchmark_result.avg_word_accuracy,
                            total_chars_reference=benchmark_result.total_chars_reference,
                            total_chars_correct=benchmark_result.total_chars_correct,
                            total_chars_substituted=benchmark_result.total_chars_substituted,
                            total_chars_deleted=benchmark_result.total_chars_deleted,
                            total_chars_inserted=benchmark_result.total_chars_inserted,
                            total_words_reference=benchmark_result.total_words_reference,
                            total_words_correct=benchmark_result.total_words_correct,
                            total_words_substituted=benchmark_result.total_words_substituted,
                            total_words_deleted=benchmark_result.total_words_deleted,
                            total_words_inserted=benchmark_result.total_words_inserted,
                            document_cer=benchmark_result.document_cer,
                            document_wer=benchmark_result.document_wer,
                            document_character_accuracy=benchmark_result.document_character_accuracy,
                            document_word_accuracy=benchmark_result.document_word_accuracy,
                            confidence_high=benchmark_result.confidence_high,
                            confidence_medium=benchmark_result.confidence_medium,
                            confidence_low=benchmark_result.confidence_low,
                            lowest_cer_page=benchmark_result.lowest_cer_page,
                            highest_cer_page=benchmark_result.highest_cer_page,
                            lowest_cer=benchmark_result.lowest_cer,
                            highest_cer=benchmark_result.highest_cer,
                            pages=pydantic_pages,
                            ground_truth_metadata=benchmark_result.ground_truth_metadata,
                            total_missing_characters=benchmark_result.total_missing_characters,
                            total_inserted_characters=benchmark_result.total_inserted_characters,
                            total_deleted_characters=benchmark_result.total_deleted_characters,
                            total_missing_words=benchmark_result.total_missing_words,
                            total_inserted_words=benchmark_result.total_inserted_words,
                            total_deleted_words=benchmark_result.total_deleted_words,
                            formatting_preservation=benchmark_result.formatting_preservation,
                            structural_preservation=benchmark_result.structural_preservation,
                            overall_accuracy=benchmark_result.overall_accuracy,
                            processing_time_seconds=benchmark_result.processing_time_seconds,
                            unified_diff=benchmark_result.unified_diff,
                        )

                        # Populate legacy EvaluationResult for backward compat
                        evaluation = EvaluationResult(
                            character_accuracy=benchmark_result.document_character_accuracy,
                            word_accuracy=benchmark_result.document_word_accuracy,
                            missing_characters=benchmark_result.total_chars_deleted,
                            extra_characters=benchmark_result.total_chars_inserted,
                            processing_time_seconds=benchmark_result.processing_time_seconds,
                            unified_diff=benchmark_result.unified_diff,
                        )
                        logger.info(
                            f"Benchmark computed: doc_cer={benchmark_result.document_cer:.4f}, "
                            f"doc_char_acc={benchmark_result.document_character_accuracy:.4f}, "
                            f"pages={len(pydantic_pages)}"
                        )
                    except Exception as eval_exc:
                        logger.error(f"Benchmark evaluation failed: {eval_exc}", exc_info=True)
                        warnings_list.append(f"Benchmark evaluation error: {eval_exc}")

                # 4. Run summary
                total_duration = time.perf_counter() - total_started_at
                avg_time = sum(p.processing_time_seconds for p in pages) / len(pages) if pages else 0.0
                avg_size = sum(p.metrics.image_size_bytes for p in pages if p.metrics) / len(pages) if pages else 0.0

                run_summary = OCRRunSummary(
                    pages_processed=len(pages),
                    pages_succeeded=len(successful_pages),
                    pages_failed=len(pages) - len(successful_pages),
                    avg_page_processing_time_seconds=round(avg_time, 6),
                    total_processing_time_seconds=round(total_duration, 6),
                    avg_image_size_bytes=round(avg_size, 2),
                    failures=failures_list,
                    warnings=warnings_list,
                )

                final_response = OCRValidationResponse(
                    pages=pages,
                    total_processing_time_seconds=round(total_duration, 6),
                    evaluation=evaluation,
                    benchmark=benchmark,
                    run_summary=run_summary,
                )

                yield f"data: {json.dumps({'type': 'result', 'result': final_response.model_dump()})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'category': 'unknown_exception', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class DownloadDocumentRequest(BaseModel):
    elements: list[DocumentElement]
    institution_name: str = "TITUS SOLUTIONS EXAM LAB"
    subject: str | None = None
    class_grade: str | None = None
    total_marks: int | None = None
    time_allowed: str | None = None
    notes: str | None = None
    answers_markdown: str | None = None


@router.post("/download-html")
async def download_html(req: DownloadDocumentRequest):
    from app.core.html_generator import create_html_from_elements
    import io

    html_output = create_html_from_elements(
        page_elements=[(1, req.elements)],
        institution_name=req.institution_name,
        subject=req.subject,
        class_grade=req.class_grade,
        total_marks=req.total_marks,
        time_allowed=req.time_allowed,
        notes=req.notes,
        title=req.subject or "TITUS Validation Export",
    )

    file_stream = io.BytesIO(html_output.encode("utf-8"))
    file_stream.seek(0)

    filename = "extracted-exam.html"
    if req.subject:
        filename = f"{req.subject.replace(' ', '_')}-exam.html"

    return StreamingResponse(
        file_stream,
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.post("/download-docx", deprecated=True)
async def download_docx(_: DownloadDocumentRequest):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="DOCX rendering is deprecated and unavailable in the production workflow. Use /download-html.",
    )


class GenerateAnswersRequest(BaseModel):
    elements: list[DocumentElement]


@router.post("/generate-answers")
async def generate_answers(req: GenerateAnswersRequest):
    from app.ocr.answer_generator import AnswerGenerator
    try:
        generator = AnswerGenerator()
        answers_md = generator.generate_answers(req.elements)
        return {"answers_markdown": answers_md}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answers: {str(exc)}"
        )


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_VALIDATION_HTML = r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>TITUS-082 OCR Validation</title>
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: #f4f6f8;
        color: #18212c;
        font-family: Arial, sans-serif;
      }
      main {
        width: min(1440px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 28px 0 48px;
      }
      h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }
      h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
      p { margin: 0; color: #576575; }
      .topbar {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-end;
        margin-bottom: 20px;
      }
      .badge {
        border: 1px solid #b7c2d0;
        border-radius: 6px;
        color: #384858;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
      }
      form {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        align-items: end;
        background: #ffffff;
        border: 1px solid #d8dee7;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
      }
      .form-full-width {
        grid-column: 1 / -1;
      }
      .form-actions {
        grid-column: 1 / -1;
        display: flex;
        justify-content: flex-end;
        margin-top: 8px;
      }
      label {
        display: grid;
        gap: 6px;
        color: #344456;
        font-size: 13px;
        font-weight: 700;
      }
      input[type="text"], input[type="number"], input[type="file"] {
        border: 1px solid #c6ced8;
        border-radius: 6px;
        padding: 9px;
        background: #fbfcfd;
        font: inherit;
        font-weight: 400;
        width: 100%;
      }
      button {
        border: 0;
        border-radius: 6px;
        background: #126b58;
        color: #ffffff;
        cursor: pointer;
        font: inherit;
        font-weight: 700;
        min-height: 40px;
        padding: 0 20px;
        transition: background 0.2s, opacity 0.2s;
      }
      button:hover:not(:disabled) {
        background: #0e5243;
      }
      button.secondary {
        background: #344456;
      }
      button.secondary:hover:not(:disabled) {
        background: #24303d;
      }
      button:disabled {
        background: #cbd5e1 !important;
        color: #64748b !important;
        cursor: not-allowed;
        opacity: 0.85;
      }
      #status {
        min-height: 24px;
        margin-bottom: 14px;
        font-weight: 700;
      }
      .error { color: #a32620; }
      .metrics {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 18px;
      }
      .metric, .panel, .page-card {
        border: 1px solid #d8dee7;
        border-radius: 8px;
        background: #ffffff;
      }
      .metric { padding: 12px; }
      .metric strong { display: block; font-size: 20px; margin-top: 4px; }
      .page-card {
        display: grid;
        grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1fr) minmax(320px, 1fr);
        gap: 0;
        margin-bottom: 16px;
        overflow: hidden;
      }
      .panel {
        border: 0;
        border-right: 1px solid #d8dee7;
        border-radius: 0;
        min-width: 0;
        padding: 14px;
      }
      .panel:last-child { border-right: 0; }
      .panel img {
        display: block;
        width: 100%;
        max-height: 720px;
        object-fit: contain;
        background: #eef1f5;
        border: 1px solid #d8dee7;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-family: Consolas, "Liberation Mono", monospace;
        font-size: 13px;
        line-height: 1.45;
      }
      .downloads {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 14px 0 18px;
      }
      .diff {
        padding: 14px;
        margin-bottom: 18px;
      }
      .diff-line {
        display: block;
        padding: 2px 4px;
      }
      .diff-add { background: #fff3a3; }
      .diff-del { background: #ffd7d7; }
      .diff-meta { color: #6b7787; }
      
      /* New CSS styles for Progress Panel and OCR Run Summary */
      .progress-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: #ffffff;
        border: 1px solid #d8dee7;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 14px;
      }
      .progress-row .badge-waiting {
        color: #6b7787;
        font-weight: bold;
      }
      .progress-row .badge-running {
        color: #1a73e8;
        font-weight: bold;
        animation: pulse 1.5s infinite;
      }
      .progress-row .badge-completed {
        color: #126b58;
        font-weight: bold;
      }
      .progress-row .badge-failed {
        color: #a32620;
        font-weight: bold;
      }
      @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
      }
      
      .summary-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }
      .summary-card {
        background: #ffffff;
        border: 1px solid #d8dee7;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
      }
      .summary-card span {
        display: block;
        font-size: 11px;
        color: #576575;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
      }
      .summary-card strong {
        font-size: 22px;
        color: #18212c;
      }
      
      .summary-alerts {
        display: grid;
        gap: 8px;
        margin-top: 14px;
      }
      .alert-box {
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.4;
      }
      .alert-failure {
        background: #fdf2f2;
        border: 1px solid #fde8e8;
        color: #9b1c1c;
      }
      .alert-warning {
        background: #fffdf2;
        border: 1px solid #fdf6b2;
        color: #723b13;
      }

      @media (max-width: 1100px) {
        form { grid-template-columns: 1fr; }
        .metrics { grid-template-columns: 1fr 1fr; }
        .page-card { grid-template-columns: 1fr; }
        .panel { border-right: 0; border-bottom: 1px solid #d8dee7; }
        .panel:last-child { border-bottom: 0; }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="topbar">
        <div>
          <h1>TITUS-082 OCR Validation</h1>
          <p>Developer-only tool for OCR subsystem accuracy checks before database integration.</p>
        </div>
        <div class="badge">DEV TOOL ONLY</div>
      </div>

      <form id="validationForm">
        <label>
          Institution Name
          <input id="institutionName" name="institution_name" type="text" value="TITUS SOLUTIONS EXAM LAB" required />
        </label>
        <label>
          Subject (required)
          <input id="subject" name="subject" type="text" placeholder="e.g. English, Math" required />
        </label>
        <label>
          Class/Grade (required)
          <input id="classGrade" name="class_grade" type="text" placeholder="e.g. Class 10, Grade A" required />
        </label>
        <label>
          Total Marks (required)
          <input id="totalMarks" name="total_marks" type="number" placeholder="e.g. 100" required />
        </label>
        <label>
          Time Allowed (optional)
          <input id="timeAllowed" name="time_allowed" type="text" placeholder="e.g. 3 Hours" />
        </label>
        <label class="form-full-width">
          Job Notes (optional)
          <input id="notes" name="notes" type="text" placeholder="e.g. Midterm exam paper" />
        </label>
        <label>
          Handwritten PDFs / images
          <input id="documentFiles" name="files" type="file" multiple required accept="application/pdf,image/jpeg,image/png,image/webp,image/tiff" />
        </label>
        <label>
          Ground truth (.txt, .pdf) (optional)
          <input id="groundTruth" name="ground_truth" type="file" accept="text/plain,.txt,application/pdf,.pdf" />
        </label>
        <div class="form-actions">
          <button id="runButton" type="submit" disabled>Run OCR Validation</button>
        </div>
      </form>

      <div id="status" aria-live="polite"></div>

      <!-- Real-time Progress Tracking Panel -->
      <section id="progressPanel" class="panel" style="display: none; border: 1px solid #d8dee7; border-radius: 8px; background: #ffffff; padding: 16px; margin-bottom: 18px; width: 100%;">
        <h2 style="margin-top: 0;">OCR Execution Progress</h2>
        <div id="progressStatus" style="font-weight: bold; margin-bottom: 12px; color: #344456;"></div>
        <div id="progressPages" style="max-height: 300px; overflow-y: auto; padding: 8px 4px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;"></div>
      </section>

      <!-- Run Summary Section -->
      <section id="runSummaryPanel" class="panel" style="display: none; border: 1px solid #d8dee7; border-radius: 8px; background: #ffffff; padding: 16px; margin-bottom: 18px; width: 100%;">
        <h2 style="margin-top: 0;">OCR Run Summary</h2>
        <div class="summary-container" id="summaryContainer"></div>
        <div class="summary-alerts" id="summaryAlerts"></div>
      </section>

      <!-- Answer Key Preview Panel (ANS-04) -->
      <section id="answerKeyPanel" class="panel" style="display: none; border: 1px solid #d8dee7; border-radius: 8px; background: #ffffff; padding: 16px; margin-bottom: 18px; width: 100%;">
        <h2 style="margin-top: 0; color: #126b58;">AI Generated Answer Key Preview</h2>
        <div id="answerKeyContent" style="padding: 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-family: Arial, sans-serif; white-space: pre-wrap; line-height: 1.5; font-size: 14px; max-height: 400px; overflow-y: auto;"></div>
      </section>

      <section id="metrics" class="metrics"></section>
      <section id="downloads" class="downloads"></section>
      <section id="diff"></section>
      <section id="pages"></section>
    </main>

    <script>
      const form = document.querySelector("#validationForm");
      const runButton = document.querySelector("#runButton");
      const statusEl = document.querySelector("#status");
      const metricsEl = document.querySelector("#metrics");
      const downloadsEl = document.querySelector("#downloads");
      const diffEl = document.querySelector("#diff");
      const pagesEl = document.querySelector("#pages");
      
      const progressPanel = document.querySelector("#progressPanel");
      const progressStatus = document.querySelector("#progressStatus");
      const progressPages = document.querySelector("#progressPages");
      const runSummaryPanel = document.querySelector("#runSummaryPanel");
      const summaryContainer = document.querySelector("#summaryContainer");
      const summaryAlerts = document.querySelector("#summaryAlerts");
      const answerKeyPanel = document.querySelector("#answerKeyPanel");
      const answerKeyContent = document.querySelector("#answerKeyContent");

      let latestResult = null;
      let generatedAnswers = null;

      const documentFiles = document.querySelector("#documentFiles");
      const institutionName = document.querySelector("#institutionName");
      const subject = document.querySelector("#subject");
      const classGrade = document.querySelector("#classGrade");
      const totalMarks = document.querySelector("#totalMarks");

      function checkFormValidity() {
        const hasFiles = documentFiles.files.length > 0;
        const instVal = institutionName.value.trim();
        const subVal = subject.value.trim();
        const gradeVal = classGrade.value.trim();
        const marksVal = totalMarks.value.trim();
        
        const isValid = hasFiles && instVal && subVal && gradeVal && marksVal;
        runButton.disabled = !isValid;
      }

      documentFiles.addEventListener("change", checkFormValidity);
      institutionName.addEventListener("input", checkFormValidity);
      subject.addEventListener("input", checkFormValidity);
      classGrade.addEventListener("input", checkFormValidity);
      totalMarks.addEventListener("input", checkFormValidity);
      
      checkFormValidity();

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData();
        for (const file of document.querySelector("#documentFiles").files) {
          formData.append("files", file);
        }
        const truth = document.querySelector("#groundTruth").files[0];
        if (truth) formData.append("ground_truth", truth);

        clearOutput();
        setStatus("Running OCR pipeline...");
        runButton.disabled = true;
        
        progressPanel.style.display = "block";
        progressStatus.textContent = "Connecting to server...";
        progressPages.replaceChildren();

        try {
          const response = await fetch("/dev/ocr-validation/process", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const errText = await response.text();
            let errMsg = "Validation failed.";
            try {
              const errJson = JSON.parse(errText);
              errMsg = errJson.detail || errMsg;
            } catch (_) {}
            throw new Error(errMsg);
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith("data: ")) {
                const dataStr = trimmed.substring(6);
                const ev = JSON.parse(dataStr);
                
                if (ev.type === "status") {
                  progressStatus.textContent = ev.message;
                } else if (ev.type === "init_pages") {
                  progressPages.replaceChildren();
                  for (let i = 1; i <= ev.total_pages; i++) {
                    const row = document.createElement("div");
                    row.id = `progress-row-${i}`;
                    row.className = "progress-row";
                    row.innerHTML = `<span>Page ${i}/${ev.total_pages}</span><span class="badge-waiting">Waiting...</span>`;
                    progressPages.append(row);
                  }
                } else if (ev.type === "page_status") {
                  const row = document.getElementById(`progress-row-${ev.page}`);
                  if (row) {
                    const badge = row.querySelector("span:last-child");
                    if (ev.status === "running") {
                      badge.className = "badge-running";
                      badge.textContent = "Running... ";
                    } else if (ev.status === "completed") {
                      badge.className = "badge-completed";
                      badge.textContent = `✓ Completed (${ev.duration.toFixed(1)} s)`;
                    } else {
                      badge.className = "badge-failed";
                      badge.textContent = `✗ ${ev.status.toUpperCase()} (${ev.reason || "Error occurred"})`;
                    }
                  }
                } else if (ev.type === "result") {
                  latestResult = ev.result;
                  renderResult(ev.result);
                  setStatus(`Processed ${ev.result.pages.length} page(s).`);
                  progressStatus.textContent = "Done.";
                } else if (ev.type === "error") {
                  throw new Error(`[${ev.category}] ${ev.message}`);
                }
              }
            }
          }
        } catch (error) {
          setStatus(error.message || "Validation failed.", true);
          progressStatus.textContent = "Failed.";
        } finally {
          runButton.disabled = false;
        }
      });

      function clearOutput() {
        latestResult = null;
        generatedAnswers = null;
        metricsEl.replaceChildren();
        downloadsEl.replaceChildren();
        diffEl.replaceChildren();
        pagesEl.replaceChildren();
        progressPanel.style.display = "none";
        runSummaryPanel.style.display = "none";
        answerKeyPanel.style.display = "none";
        answerKeyContent.replaceChildren();
      }

      function setStatus(message, isError = false) {
        statusEl.textContent = message;
        statusEl.classList.toggle("error", isError);
      }

      function renderResult(result) {
        renderMetrics(result);
        renderRunSummary(result.run_summary);
        renderDownloads(result);
        renderDiff(result.evaluation?.unified_diff || "");
        renderPages(result.pages);
      }

      function renderMetrics(result) {
        const metrics = [
          ["Total time", `${result.total_processing_time_seconds.toFixed(3)}s`],
        ];
        if (result.evaluation) {
          metrics.push(
            ["Character accuracy", `${(result.evaluation.character_accuracy * 100).toFixed(2)}%`],
            ["Word accuracy", `${(result.evaluation.word_accuracy * 100).toFixed(2)}%`],
            ["Missing chars", String(result.evaluation.missing_characters)],
            ["Extra chars", String(result.evaluation.extra_characters)],
          );
        }

        for (const [label, value] of metrics) {
          const div = document.createElement("div");
          div.className = "metric";
          div.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
          metricsEl.append(div);
        }
      }

      function renderRunSummary(summary) {
        if (!summary) {
          runSummaryPanel.style.display = "none";
          return;
        }
        
        runSummaryPanel.style.display = "block";
        summaryContainer.replaceChildren();
        summaryAlerts.replaceChildren();
        
        const cards = [
          ["Pages Processed", summary.pages_processed],
          ["Pages Succeeded", summary.pages_succeeded],
          ["Pages Failed", summary.pages_failed],
          ["Avg Processing Time", `${summary.avg_page_processing_time_seconds.toFixed(2)}s`],
          ["Total Processing Time", `${summary.total_processing_time_seconds.toFixed(2)}s`],
          ["Avg Image Size", `${(summary.avg_image_size_bytes / 1024 / 1024).toFixed(2)} MB`]
        ];
        
        for (const [label, val] of cards) {
          const card = document.createElement("div");
          card.className = "summary-card";
          card.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(val)}</strong>`;
          summaryContainer.append(card);
        }
        
        // Render failures list
        if (summary.failures && summary.failures.length > 0) {
          const div = document.createElement("div");
          div.className = "alert-box alert-failure";
          div.innerHTML = `<strong>Failures:</strong><ul style="margin: 4px 0 0 16px; padding: 0;">` +
            summary.failures.map(f => `<li>${escapeHtml(f)}</li>`).join("") +
            `</ul>`;
          summaryAlerts.append(div);
        }
        
        // Render warnings list
        if (summary.warnings && summary.warnings.length > 0) {
          const div = document.createElement("div");
          div.className = "alert-box alert-warning";
          div.innerHTML = `<strong>Warnings:</strong><ul style="margin: 4px 0 0 16px; padding: 0;">` +
            summary.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("") +
            `</ul>`;
          summaryAlerts.append(div);
        }
      }

      function renderDownloads(result) {
        downloadsEl.replaceChildren();
        addDownload("Download OCR Markdown", "ocr-markdown.md", result.pages.filter(p => p.status === "Completed").map((page) => page.markdown).join("\n\n"));
        addDownload("Download Parsed JSON", "parsed-document.json", JSON.stringify({ pages: result.pages.map(({ image_data_url, ...page }) => page) }, null, 2));
        addDownload("Download Evaluation Report", "evaluation-report.json", JSON.stringify(result.evaluation || {}, null, 2));
        
        if (!generatedAnswers) {
          addHtmlDownloadButton(result, "Download HTML Document (.html)", false);
          addGenerateAnswersButton(result);
        } else {
          addHtmlDownloadButton(result, "Download Question Paper (.html)", false);
          addHtmlDownloadButton(result, "Download Q&A Combined (.html)", true);
        }
      }

      function addHtmlDownloadButton(result, label, includeAnswers) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "secondary";
        button.textContent = label;
        button.addEventListener("click", async () => {
          const allElements = [];
          for (const page of result.pages) {
            if (page.status === "Completed" && page.elements) {
              allElements.push(...page.elements);
            }
          }
          
          const payload = {
            elements: allElements,
            institution_name: document.querySelector("#institutionName").value.trim() || "TITUS SOLUTIONS EXAM LAB",
            subject: document.querySelector("#subject").value.trim() || null,
            class_grade: document.querySelector("#classGrade").value.trim() || null,
            total_marks: parseInt(document.querySelector("#totalMarks").value.trim()) || null,
            time_allowed: document.querySelector("#timeAllowed").value.trim() || null,
            notes: document.querySelector("#notes").value.trim() || null,
            answers_markdown: includeAnswers ? generatedAnswers : null
          };
          
          try {
            button.disabled = true;
            const res = await fetch("/dev/ocr-validation/download-html", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error("HTML generation failed.");
            
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = url;
            
            let filename = "extracted-exam.html";
            if (payload.subject) {
              const baseName = payload.subject.replace(/\s+/g, "_");
              filename = includeAnswers ? `${baseName}-exam-with-answers.html` : `${baseName}-exam.html`;
            } else {
              filename = includeAnswers ? "extracted-exam-with-answers.html" : "extracted-exam.html";
            }
            
            anchor.download = filename;
            anchor.click();
            URL.revokeObjectURL(url);
          } catch (err) {
            alert(err.message || "Failed to download HTML document.");
          } finally {
            button.disabled = false;
          }
        });
        downloadsEl.append(button);
      }

      function addGenerateAnswersButton(result) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Generate Answers";
        button.style.backgroundColor = "#1a73e8";
        button.style.color = "#ffffff";
        
        button.addEventListener("click", async () => {
          const confirmed = confirm("This will add AI-generated answers to your document. Continue?");
          if (!confirmed) return;
          
          const allElements = [];
          for (const page of result.pages) {
            if (page.status === "Completed" && page.elements) {
              allElements.push(...page.elements);
            }
          }
          
          try {
            button.disabled = true;
            setStatus("Generating answers...");
            
            const res = await fetch("/dev/ocr-validation/generate-answers", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({ elements: allElements })
            });
            
            if (!res.ok) {
              const errText = await res.text();
              throw new Error(errText || "Failed to generate answers.");
            }
            
            const data = await res.json();
            generatedAnswers = data.answers_markdown;
            
            answerKeyPanel.style.display = "block";
            answerKeyContent.textContent = generatedAnswers;
            
            renderDownloads(result);
            setStatus("Answer generation complete.");
          } catch (err) {
            alert(err.message || "Failed to generate answers.");
            setStatus("Answer generation failed.", true);
          } finally {
            button.disabled = false;
          }
        });
        downloadsEl.append(button);
      }

      function addDownload(label, filename, content) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "secondary";
        button.textContent = label;
        button.addEventListener("click", () => {
          const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
          const url = URL.createObjectURL(blob);
          const anchor = document.createElement("a");
          anchor.href = url;
          anchor.download = filename;
          anchor.click();
          URL.revokeObjectURL(url);
        });
        downloadsEl.append(button);
      }

      function renderDiff(diffText) {
        if (!diffText) return;
        const panel = document.createElement("section");
        panel.className = "panel diff";
        const title = document.createElement("h2");
        title.textContent = "Unified Diff";
        const pre = document.createElement("pre");
        for (const line of diffText.split("\n")) {
          const span = document.createElement("span");
          span.className = "diff-line";
          if (line.startsWith("+") && !line.startsWith("+++")) span.classList.add("diff-add");
          if (line.startsWith("-") && !line.startsWith("---")) span.classList.add("diff-del");
          if (line.startsWith("@@") || line.startsWith("---") || line.startsWith("+++")) span.classList.add("diff-meta");
          
          let escaped = escapeHtml(line);
          escaped = escaped.replace(
            /&lt;span\s+(?:class=["']low-confidence["']\s+data-confidence=["'](\d+)["']|data-confidence=["'](\d+)["']\s+class=["']low-confidence["'])\s*&gt;(.*?)&lt;\/span&gt;/gi,
            (match, conf1, conf2, word) => {
              const confidence = conf1 || conf2;
              return `<span class="low-confidence" style="background: yellow; border-bottom: 1px dotted red; cursor: help;" title="Confidence: ${confidence}%">${word}</span>`;
            }
          );
          span.innerHTML = escaped;
          pre.append(span);
        }
        panel.append(title, pre);
        diffEl.append(panel);
      }

      function renderPages(pages) {
        for (const page of pages) {
          const card = document.createElement("article");
          card.className = "page-card";
          
          // Left column: Page image
          const leftCol = panel(`Page ${page.page} Image`, image(page.image_data_url));
          
          // Middle column: Gemini Markdown or Error banner + metrics badge
          const midContainer = document.createElement("div");
          
          if (page.status !== "Completed") {
            const errBanner = document.createElement("div");
            errBanner.style = "padding: 10px; background: #fef2f2; border: 1px solid #fee2e2; border-radius: 6px; color: #991b1b; margin-bottom: 12px; font-size: 13px; line-height: 1.4;";
            errBanner.innerHTML = `<strong>Status:</strong> ${escapeHtml(page.status)}<br><strong>Reason:</strong> ${escapeHtml(page.error_reason || "OCR execution failed")}`;
            midContainer.append(errBanner);
          }
          
          if (page.metrics) {
            const metricsBadge = document.createElement("div");
            metricsBadge.style = "margin-bottom: 12px; padding: 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 12px; line-height: 1.6; color: #334155;";
            metricsBadge.innerHTML = `
              <div style="font-weight: bold; margin-bottom: 4px; color: #475569;">Page timing metrics:</div>
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                <div>Preprocessing: <strong>${page.metrics.preprocessing_time_seconds.toFixed(3)}s</strong></div>
                <div>Gemini request: <strong>${page.metrics.gemini_request_seconds.toFixed(3)}s</strong></div>
                <div>Parsing: <strong>${page.metrics.parsing_time_seconds.toFixed(3)}s</strong></div>
                <div>Total page time: <strong>${page.metrics.total_time_seconds.toFixed(3)}s</strong></div>
                <div>Resolution: <strong>${escapeHtml(page.metrics.image_resolution)}</strong></div>
                <div>Image size: <strong>${(page.metrics.image_size_bytes / 1024 / 1024).toFixed(2)} MB</strong></div>
              </div>
              <div style="margin-top: 4px; border-top: 1px solid #e2e8f0; padding-top: 4px;">
                Retries: <strong>${page.retry_count}</strong>
              </div>
            `;
            midContainer.append(metricsBadge);
          }
          
          if (page.status === "Completed") {
            const mdPre = preMarkdown(page.markdown);
            midContainer.append(mdPre);
          } else {
            const failedText = document.createElement("div");
            failedText.style = "color: #64748b; font-style: italic; font-size: 13px;";
            failedText.textContent = "(No markdown output generated)";
            midContainer.append(failedText);
          }
          
          const midCol = panel(`Gemini Markdown (${page.processing_time_seconds.toFixed(3)}s)`, midContainer);
          
          // Right column: Parsed JSON
          const rightContainer = document.createElement("div");
          if (page.status === "Completed") {
            rightContainer.append(pre(JSON.stringify({ ...page, image_data_url: undefined }, null, 2)));
          } else {
            rightContainer.append(pre(JSON.stringify({
              page: page.page,
              status: page.status,
              error_reason: page.error_reason,
              metrics: page.metrics,
              retry_count: page.retry_count
            }, null, 2)));
          }
          
          const rightCol = panel("Parsed Document JSON", rightContainer);
          
          card.append(leftCol, midCol, rightCol);
          pagesEl.append(card);
        }
      }

      function panel(titleText, child) {
        const section = document.createElement("section");
        section.className = "panel";
        const title = document.createElement("h2");
        title.textContent = titleText;
        section.append(title, child);
        return section;
      }

      function image(src) {
        const img = document.createElement("img");
        img.src = src;
        img.alt = "Rendered OCR page";
        return img;
      }

      function preMarkdown(text) {
        const node = document.createElement("pre");
        let escaped = escapeHtml(text);
        escaped = escaped.replace(
          /&lt;span\s+(?:class=["']low-confidence["']\s+data-confidence=["'](\d+)["']|data-confidence=["'](\d+)["']\s+class=["']low-confidence["'])\s*&gt;(.*?)&lt;\/span&gt;/gi,
          (match, conf1, conf2, word) => {
            const confidence = conf1 || conf2;
            return `<span class="low-confidence" style="background: yellow; border-bottom: 1px dotted red; cursor: help;" title="Confidence: ${confidence}%">${word}</span>`;
          }
        );
        node.innerHTML = escaped;
        return node;
      }

      function pre(text) {
        const node = document.createElement("pre");
        node.textContent = text;
        return node;
      }

      function escapeHtml(text) {
        return String(text)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;");
      }
    </script>
  </body>
</html>
"""
