import asyncio
import io
import logging
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.db import JSONDatabase
from app.core.examination_builder import (
    elements_to_structured_page,
    page_plain_text,
    page_result_to_structured_page,
    persist_structured_document,
)
from app.core.examination_model import Page as StructuredPage
from app.core.html_renderer import HTMLRenderer, html_filename
from app.core.image import ImageProcessingError, preprocess_image, normalize_page_image_with_metrics
from app.core.pdf import PDFProcessingError, split_pdf_to_images
from app.core.tempfiles import LocalTempStorage
from app.core.validator import FileValidationError, UploadedFile, validate_upload
from app.ocr.intelligence import analyze_examination_page
from app.ocr.models import DocumentElement, OCRValidationPage, PageMetrics
from app.ocr.parser import parse_markdown
from app.ocr.service import OCRService, OCRServiceError, OCRTimeoutError

logger = logging.getLogger("ocr_pipeline")
router = APIRouter(prefix="/api/v1", tags=["jobs-api"])


# Pydantic models for request/response
class PageUpdatePayload(BaseModel):
    markdown: str = ""
    elements: list[DocumentElement] = Field(default_factory=list)
    structured_page: StructuredPage | None = None


class SettingsPayload(BaseModel):
    institution_name: str
    primary_ocr_engine: str


# Helper to format timestamp
def format_relative_time(timestamp: float) -> str:
    diff = time.time() - timestamp
    if diff < 60:
        return "Just now"
    elif diff < 3600:
        return f"{int(diff // 60)} minutes ago"
    elif diff < 86400:
        return f"{int(diff // 3600)} hours ago"
    else:
        return f"{int(diff // 86400)} days ago"


def _html_export_response(job_id: str, job: dict[str, Any], db: JSONDatabase, *, details: str) -> StreamingResponse:
    document = persist_structured_document(job)
    db.save_job(job_id, job)

    html_output = HTMLRenderer().render(document)
    file_stream = io.BytesIO(html_output.encode("utf-8"))
    file_stream.seek(0)

    filename = html_filename(document.metadata)
    db.log_audit_event("Generate HTML Document", job_id, job["name"], details=details)

    return StreamingResponse(
        file_stream,
        media_type=HTMLRenderer.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


async def _run_ocr_pipeline(job_id: str, files_info: list[tuple[str, bytes]]) -> None:
    db = JSONDatabase()
    job = db.get_job(job_id)
    if not job:
        return

    storage = LocalTempStorage(settings.temp_root)
    # Setup job directories
    job_dir = settings.temp_root / f"job-{job_id}"
    uploads_dir = job_dir / "uploads"
    pages_dir = job_dir / "pages"
    
    uploads_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize storage workdir to job specific path so it doesn't use a random temp folder
    storage.workdir = job_dir

    pages_list: list[dict[str, Any]] = []
    warnings_list: list[str] = []
    failures_list: list[str] = []
    
    # Store page processing tasks
    page_tasks = []
    current_page_num = 1

    try:
        # Save uploads and extract pages
        for filename, content in files_info:
            uploaded_file = UploadedFile(
                filename=filename,
                content_type=None,
                size=len(content),
                data=content,
            )

            try:
                validated = validate_upload(uploaded_file)
            except FileValidationError as exc:
                failures_list.append(f"Validation failed for {filename}: {str(exc)}")
                continue

            # Save upload file
            saved_upload_path = uploads_dir / validated.filename
            saved_upload_path.write_bytes(validated.data)

            if validated.kind == "pdf":
                try:
                    split_pages = await split_pdf_to_images(
                        saved_upload_path,
                        storage,
                        start_page=current_page_num,
                        display_name=validated.filename,
                    )
                    # We copy split files to pages folder and change path
                    for page_img_path in split_pages:
                        final_page_path = pages_dir / f"page_{current_page_num}.png"
                        if page_img_path.exists():
                            page_img_path.replace(final_page_path)
                        page_tasks.append((current_page_num, final_page_path, validated.filename, True))
                        current_page_num += 1
                except Exception as exc:
                    failures_list.append(f"PDF processing failed for {validated.filename}: {str(exc)}")
            else:
                try:
                    # Preprocess single image
                    page_img_path = await preprocess_image(
                        saved_upload_path,
                        storage,
                        page_number=current_page_num,
                        display_name=validated.filename,
                    )
                    final_page_path = pages_dir / f"page_{current_page_num}.png"
                    if page_img_path.exists():
                        page_img_path.replace(final_page_path)
                    page_tasks.append((current_page_num, final_page_path, validated.filename, False))
                    current_page_num += 1
                except Exception as exc:
                    failures_list.append(f"Image processing failed for {validated.filename}: {str(exc)}")

        total_pages = len(page_tasks)
        if total_pages == 0:
            job["status"] = "failed"
            job["failures"] = failures_list
            db.save_job(job_id, job)
            db.log_audit_event("OCR Failed", job_id, job["name"], details="No valid pages extracted.")
            return

        job["pages_count"] = total_pages
        db.save_job(job_id, job)

        # Initialize OCR provider
        ocr_service = OCRService()

        # Run page extraction
        for page_num, page_image_path, source_filename, is_pdf in page_tasks:
            page_started_at = time.perf_counter()
            preprocessing_duration = 0.0
            image_size = 0
            image_resolution = ""
            gemini_duration = 0.0
            parsing_duration = 0.0
            retry_count = 0

            try:
                # Normalize page image
                preprocess_res = await normalize_page_image_with_metrics(
                    page_image_path, display_name=source_filename
                )
                preprocessing_duration = preprocess_res.duration_seconds
                image_size = preprocess_res.metadata.size_bytes
                image_resolution = f"{preprocess_res.metadata.width}x{preprocess_res.metadata.height}"
                warnings_list.extend(preprocess_res.warnings)

                # Execute OCR
                ocr_res = await ocr_service.extract_page(page_num, preprocess_res.path)
                gemini_duration = ocr_res.gemini_request_seconds
                retry_count = ocr_res.retry_count

                # Parse elements
                parsing_start = time.perf_counter()
                
                raw_markdown = ocr_res.markdown.strip()
                cleaned_markdown = raw_markdown
                if cleaned_markdown.startswith("```"):
                    first_nl = cleaned_markdown.find("\n")
                    if first_nl != -1:
                        cleaned_markdown = cleaned_markdown[first_nl:].strip()
                    if cleaned_markdown.endswith("```"):
                        cleaned_markdown = cleaned_markdown[:-3].strip()

                parsed_page = analyze_examination_page(page_num, cleaned_markdown)
                parsing_duration = time.perf_counter() - parsing_start

                total_page_time = time.perf_counter() - page_started_at

                # Format static URL
                static_url = f"/static/titus-082/job-{job_id}/pages/page_{page_num}.png"
                structured_page = page_result_to_structured_page(
                    parsed_page,
                    status="completed",
                    image_url=static_url,
                )

                page_result = {
                    "page_number": page_num,
                    "status": "completed",
                    "image_url": static_url,
                    "markdown": cleaned_markdown,
                    "plain_text": parsed_page.plain_text,
                    "elements": [el.model_dump() for el in parsed_page.elements],
                    "structured_page": structured_page.model_dump(),
                    "processing_time_seconds": round(total_page_time, 4),
                    "metrics": {
                        "preprocessing_time_seconds": round(preprocessing_duration, 4),
                        "image_size_bytes": image_size,
                        "image_resolution": image_resolution,
                        "gemini_request_seconds": round(gemini_duration, 4),
                        "parsing_time_seconds": round(parsing_duration, 4),
                        "total_time_seconds": round(total_page_time, 4),
                    },
                    "retry_count": retry_count,
                    "quality_report": preprocess_res.quality_report,
                }
                pages_list.append(page_result)

            except Exception as exc:
                total_page_time = time.perf_counter() - page_started_at
                logger.error(f"Error processing page {page_num} for job {job_id}: {exc}")
                
                static_url = f"/static/titus-082/job-{job_id}/pages/page_{page_num}.png"
                page_result = {
                    "page_number": page_num,
                    "status": "failed",
                    "error_reason": str(exc),
                    "image_url": static_url,
                    "markdown": "",
                    "plain_text": "",
                    "elements": [],
                    "structured_page": elements_to_structured_page(
                        page_number=page_num,
                        elements=[],
                        status="failed",
                        image_url=static_url,
                    ).model_dump(),
                    "processing_time_seconds": round(total_page_time, 4),
                    "retry_count": retry_count,
                    "quality_report": preprocess_res.quality_report if 'preprocess_res' in locals() else None,
                }
                pages_list.append(page_result)
                failures_list.append(f"Page {page_num}: {exc}")

            # Update pages processed progress
            job["pages_processed"] = len(pages_list)
            db.save_job(job_id, job)

        # Decide final job status
        has_failures = any(p["status"] == "failed" for p in pages_list)
        job["status"] = "pending_review" if (has_failures or len(pages_list) > 0) else "failed"
        job["pages"] = pages_list
        job["failures"] = failures_list
        job["warnings"] = warnings_list
        persist_structured_document(job)
        db.save_job(job_id, job)

        db.log_audit_event("OCR Complete", job_id, job["name"], details=f"Successfully extracted {len(pages_list)} pages.")

    except Exception as exc:
        logger.error(f"Critical error in job background process {job_id}: {exc}")
        job["status"] = "failed"
        job["failures"] = failures_list + [str(exc)]
        db.save_job(job_id, job)
        db.log_audit_event("OCR Failed", job_id, job["name"], details=str(exc))


# Endpoints

@router.get("/jobs")
def list_jobs():
    db = JSONDatabase()
    jobs = db.list_jobs()
    
    # Map job records to dashboard expectations
    result = []
    for j in jobs:
        result.append({
            "id": j["id"],
            "name": j["name"],
            "subject": j.get("metadata", {}).get("subject") or "N/A",
            "category": j.get("metadata", {}).get("category") or "N/A",
            "pages": j.get("pages_count", 0),
            "date": format_relative_time(j.get("created_at", 0)),
            "status": j["status"],
            "pages_processed": j.get("pages_processed", 0)
        })
    return result


class LocalPathValidationRequest(BaseModel):
    paths: list[str]


@router.post("/jobs/validate-local-path")
def validate_local_paths(payload: LocalPathValidationRequest):
    valid_files = []
    invalid_paths = []
    
    for path_str in payload.paths:
        path_str = path_str.strip()
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists() or not p.is_file():
            invalid_paths.append(path_str)
            continue
            
        ext = p.suffix.lower()
        if ext not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"}:
            invalid_paths.append(f"{path_str} (Unsupported format)")
            continue
            
        valid_files.append({
            "name": p.name,
            "size": p.stat().st_size,
            "path": str(p.resolve())
        })
        
    if invalid_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unsupported file paths: {', '.join(invalid_paths)}"
        )
        
    return {"files": valid_files}


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(default=[]),
    local_paths: str = Form(""),
    title: str = Form(""),
    category: str = Form(""),
    subject: str = Form(""),
    language: str = Form("English"),
    classGrade: str = Form(""),
    maxMarks: str = Form(""),
    timeDuration: str = Form(""),
    instructions: str = Form(""),
    notes: str = Form(""),
    institutionName: str = Form("TITUS SOLUTIONS EXAM LAB"),
):
    if not files and not local_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file upload or local file path is required."
        )

    db = JSONDatabase()
    job_id = secrets.token_hex(4)
    
    if files:
        primary_name = files[0].filename or "Document"
    elif local_paths:
        first_path = local_paths.split(",")[0].strip()
        primary_name = Path(first_path).name or "Document"
    else:
        primary_name = "Document"

    metadata = {
        "title": title or primary_name,
        "category": category or "Other",
        "subject": subject,
        "language": language,
        "classGrade": classGrade,
        "maxMarks": maxMarks,
        "timeDuration": timeDuration,
        "instructions": instructions,
        "notes": notes,
        "institutionName": institutionName
    }

    job_data = {
        "id": job_id,
        "name": title or primary_name,
        "metadata": metadata,
        "status": "processing",
        "created_at": time.time(),
        "pages_count": 0,
        "pages_processed": 0,
        "pages": [],
        "failures": [],
        "warnings": []
    }

    db.save_job(job_id, job_data)
    
    num_files = len(files) + (len([p for p in local_paths.split(",") if p.strip()]) if local_paths else 0)
    db.log_audit_event("Upload Document", job_id, job_data["name"], details=f"Uploaded {num_files} files.")

    # Read file content for background processing
    files_info = []
    for upload in files:
        content = await upload.read()
        files_info.append((upload.filename or "upload", content))
        
    if local_paths:
        for path_str in local_paths.split(","):
            path_str = path_str.strip()
            if not path_str:
                continue
            p = Path(path_str)
            if p.exists() and p.is_file():
                content = p.read_bytes()
                files_info.append((p.name, content))

    background_tasks.add_task(_run_ocr_pipeline, job_id, files_info)

    return {"jobId": job_id, "status": "processing"}


@router.get("/jobs/{job_id}")
def get_job_details(job_id: str):
    db = JSONDatabase()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    # Ensure backward compatibility for pages missing quality_report
    if "pages" in job:
        for page in job["pages"]:
            if "quality_report" not in page:
                page["quality_report"] = {
                    "resolution": {"status": "green", "label": "Good", "value": "Unknown"},
                    "blur": {"status": "green", "label": "Good", "value": "Unknown"},
                    "contrast": {"status": "green", "label": "Good", "value": "Unknown"},
                    "noise": {"status": "green", "label": "Low", "value": "Unknown"},
                    "perspective": {"status": "green", "label": "Good", "value": "Unknown"},
                    "exposure": {"status": "green", "label": "Good", "value": "Unknown"},
                    "jpeg_blockiness": {"status": "green", "label": "Good", "value": "Unknown"},
                    "difficulty": "Low",
                    "overall_status": "Good",
                    "recommendations": []
                }
    if "pages" in job:
        persist_structured_document(job)
        db.save_job(job_id, job)
    return job


@router.put("/jobs/{job_id}/pages/{page_num}")
def update_page_text(job_id: str, page_num: int, payload: PageUpdatePayload):
    db = JSONDatabase()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    page_idx = -1
    for idx, page in enumerate(job.get("pages", [])):
        if page["page_number"] == page_num:
            page_idx = idx
            break

    if page_idx == -1:
        raise HTTPException(status_code=404, detail="Page not found inside job.")

    page_record = job["pages"][page_idx]
    page_record["markdown"] = payload.markdown

    elements = payload.elements
    if payload.structured_page is not None:
        structured_page = payload.structured_page
        page_record["structured_page"] = structured_page.model_dump()
        page_record["plain_text"] = page_plain_text(structured_page)
    else:
        if not elements:
            parsed = parse_markdown(page_num, payload.markdown)
            elements = parsed.elements

        structured_page = elements_to_structured_page(
            page_number=page_num,
            elements=elements,
            status=page_record.get("status", "completed"),
            image_url=page_record.get("image_url"),
        )
        page_record["structured_page"] = structured_page.model_dump()
        page_record["plain_text"] = page_plain_text(structured_page)

    if elements:
        page_record["elements"] = [el.model_dump() for el in elements]

    # If all pages are completed/reviewed, status can transition to 'completed'
    persist_structured_document(job)
    db.save_job(job_id, job)
    db.log_audit_event("Edit Structured Model", job_id, job["name"], details=f"Edited structured model on Page {page_num}.")
    
    return {"status": "success"}


@router.post("/jobs/{job_id}/complete")
def finalize_job(job_id: str):
    db = JSONDatabase()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    job["status"] = "completed"
    db.log_audit_event("Finalize Job", job_id, job["name"], details="Marked review as complete.")
    return _html_export_response(
        job_id,
        job,
        db,
        details="Downloaded standalone .html output file from review completion.",
    )


@router.get("/jobs/{job_id}/download")
def download_job_html(job_id: str):
    db = JSONDatabase()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return _html_export_response(
        job_id,
        job,
        db,
        details="Downloaded standalone .html output file.",
    )


@router.get("/audit-logs")
def get_audit_logs():
    db = JSONDatabase()
    return db.get_audit_logs()


@router.get("/settings")
def get_global_settings():
    return {
        "institution_name": "TITUS SOLUTIONS EXAM LAB",
        "primary_ocr_engine": settings.primary_ocr_engine
    }


@router.put("/settings")
def update_global_settings(payload: SettingsPayload):
    # Simply log the settings change
    db = JSONDatabase()
    db.log_audit_event(
        "Update Settings", 
        "system", 
        "System Settings", 
        details=f"Set Institution to '{payload.institution_name}', Engine to '{payload.primary_ocr_engine}'."
    )
    return {"status": "success"}


@router.delete("/jobs/{job_id}")
def delete_job_route(job_id: str):
    db = JSONDatabase()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    db.delete_job(job_id)
    db.log_audit_event("Delete Job", job_id, job["name"], details="Deleted document from registry.")
    return {"status": "success"}
