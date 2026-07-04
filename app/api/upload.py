from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import settings
from app.core.image import ImageProcessingError, preprocess_image
from app.core.pdf import PDFProcessingError, split_pdf_to_images
from app.core.tempfiles import LocalTempStorage
from app.core.validator import (
    FileValidationError,
    UploadedFile,
    validate_upload,
)
from app.ocr.intelligence import analyze_examination_page
from app.ocr.models import UploadResponse
from app.ocr.service import OCRService, OCRServiceError, OCRTimeoutError

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )

    storage = LocalTempStorage(settings.temp_root)
    pages = []
    ocr_service: OCRService | None = None

    try:
        async with storage:
            page_number = 1
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
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(exc),
                    ) from exc

                stored_upload = await storage.save_upload(
                    validated.filename,
                    validated.data,
                )

                if validated.kind == "pdf":
                    try:
                        page_images = await split_pdf_to_images(
                            stored_upload,
                            storage,
                            start_page=page_number,
                            display_name=validated.filename,
                        )
                    except PDFProcessingError as exc:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc),
                        ) from exc
                else:
                    try:
                        page_image = await preprocess_image(
                            stored_upload,
                            storage,
                            page_number=page_number,
                            display_name=validated.filename,
                        )
                    except ImageProcessingError as exc:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc),
                        ) from exc
                    page_images = [page_image]

                if page_number + len(page_images) - 1 > settings.max_pages_per_job:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            "A single job can contain up to "
                            f"{settings.max_pages_per_job} pages."
                        ),
                    )

                for page_image in page_images:
                    if ocr_service is None:
                        try:
                            ocr_service = OCRService()
                        except RuntimeError as exc:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=str(exc),
                            ) from exc

                    try:
                        markdown = await ocr_service.extract_page_markdown(
                            page_number=page_number,
                            image_path=page_image,
                        )
                    except OCRTimeoutError as exc:
                        raise HTTPException(
                            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail=str(exc),
                        ) from exc
                    except OCRServiceError as exc:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=str(exc),
                        ) from exc

                    pages.append(analyze_examination_page(page_number, markdown))
                    page_number += 1

        return UploadResponse(pages=pages)
    finally:
        await storage.cleanup()
