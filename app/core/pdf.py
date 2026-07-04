import asyncio
from pathlib import Path

import pypdfium2 as pdfium

from app.config import settings
from app.core.tempfiles import LocalTempStorage


class PDFProcessingError(RuntimeError):
    pass


async def split_pdf_to_images(
    pdf_path: Path,
    storage: LocalTempStorage,
    start_page: int,
    display_name: str | None = None,
) -> list[Path]:
    try:
        return await asyncio.to_thread(
            _split_pdf_to_images_sync,
            pdf_path,
            storage,
            start_page,
            display_name or pdf_path.name,
        )
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            f'Could not process PDF "{display_name or pdf_path.name}".'
        ) from exc


def _split_pdf_to_images_sync(
    pdf_path: Path,
    storage: LocalTempStorage,
    start_page: int,
    display_name: str,
) -> list[Path]:
    try:
        document = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise PDFProcessingError(f'PDF "{display_name}" is corrupted or unreadable.') from exc

    page_count = len(document)
    if page_count == 0:
        raise PDFProcessingError(f'PDF "{display_name}" contains no pages.')
    if page_count > settings.max_pages_per_job:
        raise PDFProcessingError(
            f'PDF "{display_name}" has {page_count} pages; the PRD limit is '
            f"{settings.max_pages_per_job} pages per job."
        )

    output_paths: list[Path] = []
    try:
        for index in range(page_count):
            page_number = start_page + index
            page = document[index]
            bitmap = page.render(scale=settings.pdf_render_scale)
            image = bitmap.to_pil().convert("RGB")
            output_path = storage._require_workdir() / "pages" / f"page-{page_number:04d}.png"
            image.save(output_path, format="PNG", optimize=True)
            output_paths.append(output_path)
            page.close()
    except Exception as exc:
        raise PDFProcessingError(
            f'Failed while rendering pages from PDF "{display_name}".'
        ) from exc
    finally:
        document.close()

    return output_paths
