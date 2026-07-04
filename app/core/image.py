import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import logging
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.core.tempfiles import LocalTempStorage

logger = logging.getLogger("ocr_pipeline")


class ImageProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageMetadata:
    original_width: int
    original_height: int
    width: int
    height: int
    size_bytes: int
    resized: bool
    estimated_dpi: float


@dataclass(frozen=True)
class ImagePreprocessResult:
    path: Path
    metadata: ImageMetadata
    duration_seconds: float
    warnings: list[str]
    quality_report: dict | None = None


async def preprocess_image(
    source_path: Path,
    storage: LocalTempStorage,
    page_number: int = 1,
    display_name: str | None = None,
) -> Path:
    result = await preprocess_image_with_metrics(
        source_path=source_path,
        storage=storage,
        page_number=page_number,
        display_name=display_name,
    )
    return result.path


async def preprocess_image_with_metrics(
    source_path: Path,
    storage: LocalTempStorage,
    page_number: int = 1,
    display_name: str | None = None,
) -> ImagePreprocessResult:
    destination = await storage.reserve_page_path(page_number)
    return await asyncio.to_thread(
        _preprocess_image_sync,
        source_path,
        destination,
        display_name or source_path.name,
    )


async def normalize_page_image_with_metrics(
    source_path: Path,
    display_name: str | None = None,
) -> ImagePreprocessResult:
    return await asyncio.to_thread(
        _preprocess_image_sync,
        source_path,
        source_path,
        display_name or source_path.name,
    )


def _preprocess_image_sync(
    source_path: Path,
    destination: Path,
    display_name: str,
) -> ImagePreprocessResult:
    started_at = perf_counter()
    warnings: list[str] = []

    try:
        with Image.open(source_path) as raw_image:
            original_width, original_height = raw_image.size
            image = ImageOps.exif_transpose(raw_image)
            image = image.convert("RGB")
            
            # Estimate DPI (kept for backward compatibility metadata)
            estimated_dpi = (image.width / 8.27 + image.height / 11.69) / 2
            dpi_meta = image.info.get("dpi")
            if dpi_meta and isinstance(dpi_meta, tuple) and len(dpi_meta) >= 2:
                estimated_dpi = float(dpi_meta[0])

            image, resized = _resize_if_excessively_large(image)
            image.save(destination, format="PNG", optimize=True)

        size_bytes = destination.stat().st_size
        if resized:
            logger.info(
                f"Resizing image: Original resolution {original_width}x{original_height} "
                f"↓ New resolution {image.width}x{image.height}"
            )
            warnings.append(
                "Image resized from "
                f"{original_width}x{original_height} to {image.width}x{image.height}."
            )

        # Run new Image Quality Assessment (IQA) pipeline
        from app.core.iqa import analyze_image_quality
        quality_report = analyze_image_quality(destination)

        # Map quality report issues to warnings to preserve audit log warnings list
        if quality_report["difficulty"] != "Low":
            issues = []
            for metric_name in ["blur", "contrast", "perspective", "exposure"]:
                m = quality_report[metric_name]
                if m["status"] != "green":
                    issues.append(f"{metric_name.capitalize()}: {m['label']}")
            if issues:
                warnings.append(
                    f'"{display_name}": Quality warning — {", ".join(issues)}.'
                )

        return ImagePreprocessResult(
            path=destination,
            metadata=ImageMetadata(
                original_width=original_width,
                original_height=original_height,
                width=image.width,
                height=image.height,
                size_bytes=size_bytes,
                resized=resized,
                estimated_dpi=round(estimated_dpi, 2),
            ),
            duration_seconds=round(perf_counter() - started_at, 6),
            warnings=warnings,
            quality_report=quality_report,
        )
    except UnidentifiedImageError as exc:
        raise ImageProcessingError(f'"{display_name}" is not a readable image.') from exc
    except OSError as exc:
        raise ImageProcessingError(
            f'Image preprocessing failed for "{display_name}".'
        ) from exc


def _resize_if_excessively_large(image: Image.Image) -> tuple[Image.Image, bool]:
    max_edge = max(image.size)
    if max_edge <= settings.max_image_side:
        return image, False

    scale = settings.max_image_side / max_edge
    new_size = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS), True
