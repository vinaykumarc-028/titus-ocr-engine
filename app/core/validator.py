from dataclasses import dataclass
from pathlib import Path

from app.config import settings


class FileValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content_type: str | None
    size: int
    data: bytes


@dataclass(frozen=True)
class ValidatedUpload(UploadedFile):
    kind: str
    suffix: str


ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
}
ALLOWED_PDF_SUFFIXES = {".pdf"}


def validate_upload(file: UploadedFile) -> ValidatedUpload:
    if file.size == 0:
        raise FileValidationError(f'"{file.filename}" is empty.')

    suffix = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    if suffix in ALLOWED_PDF_SUFFIXES or content_type == "application/pdf":
        if suffix not in ALLOWED_PDF_SUFFIXES:
            raise FileValidationError("PDF files must use the .pdf extension.")
        if file.size > settings.max_pdf_bytes:
            raise FileValidationError(
                f'"{file.filename}" ({file.size / (1024 * 1024):.2f} MB) exceeds the 50 MB PDF limit.'
            )
        return ValidatedUpload(
            filename=file.filename,
            content_type="application/pdf",
            size=file.size,
            data=file.data,
            kind="pdf",
            suffix=".pdf",
        )

    if suffix in ALLOWED_IMAGE_SUFFIXES or content_type in ALLOWED_IMAGE_MIME_TYPES:
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise FileValidationError(
                "Image files must use JPG, JPEG, PNG, WEBP, or TIFF extensions."
            )
        if content_type and content_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise FileValidationError(
                f'"{file.filename}" has unsupported image MIME type {content_type}.'
            )
        if file.size > settings.max_image_bytes:
            raise FileValidationError(
                f'"{file.filename}" ({file.size / (1024 * 1024):.2f} MB) exceeds the 10 MB image limit.'
            )
        return ValidatedUpload(
            filename=file.filename,
            content_type=content_type or "application/octet-stream",
            size=file.size,
            data=file.data,
            kind="image",
            suffix=suffix,
        )

    raise FileValidationError(
        "Only JPG, JPEG, PNG, WEBP, TIFF, and PDF files are accepted."
    )
