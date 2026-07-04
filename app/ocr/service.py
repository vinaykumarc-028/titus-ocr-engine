import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.config import settings
from app.ocr.base import OCRProvider

logger = logging.getLogger("ocr_pipeline")


class OCRServiceError(RuntimeError):
    def __init__(self, message: str, category: str = "unknown_exception") -> None:
        super().__init__(message)
        self.category = category


class OCRTimeoutError(OCRServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="gemini_api_timeout")


class OCRNonRetryableError(OCRServiceError):
    pass


@dataclass(frozen=True)
class OCRRequestResult:
    markdown: str
    gemini_request_seconds: float
    retry_count: int


class OCRService:
    def __init__(self, provider: OCRProvider | None = None) -> None:
        if provider is not None:
            self._primary_provider = provider
            self._fallback_provider = None
        else:
            if settings.primary_ocr_engine == "gemini":
                from app.ocr.gemini import GeminiOCRProvider
                self._primary_provider = GeminiOCRProvider()
                if settings.mistral_api_key:
                    from app.ocr.mistral import MistralOCRProvider
                    self._fallback_provider = MistralOCRProvider()
                else:
                    self._fallback_provider = None
            else:
                from app.ocr.mistral import MistralOCRProvider
                self._primary_provider = MistralOCRProvider()
                if settings.gemini_api_key or settings.google_api_key:
                    from app.ocr.gemini import GeminiOCRProvider
                    self._fallback_provider = GeminiOCRProvider()
                else:
                    self._fallback_provider = None

    async def extract_page_markdown(self, page_number: int, image_path: Path) -> str:
        result = await self.extract_page(page_number, image_path)
        return result.markdown

    async def extract_page(self, page_number: int, image_path: Path) -> OCRRequestResult:
        attempts = max(1, settings.max_retries)
        started_at = perf_counter()
        last_error: OCRServiceError | None = None

        for attempt in range(1, attempts + 1):
            try:
                logger.info(f"Page {page_number}: OCR attempt {attempt} of {attempts}")
                markdown = await asyncio.wait_for(
                    self._primary_provider.extract_markdown(page_number, image_path),
                    timeout=settings.ocr_timeout_seconds,
                )
                return OCRRequestResult(
                    markdown=markdown,
                    gemini_request_seconds=round(perf_counter() - started_at, 6),
                    retry_count=attempt - 1,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                last_error = OCRTimeoutError(
                    f"OCR exceeded timeout for page {page_number} after "
                    f"{settings.ocr_timeout_seconds} seconds."
                )
                logger.warning(f"Page {page_number}: Attempt {attempt} timed out: {last_error}")
                if attempt == attempts:
                    break
            except Exception as exc:
                category = classify_ocr_exception(exc)
                if category in {"authentication_failure", "validation_error"}:
                    logger.error(
                        f"Page {page_number}: Attempt {attempt} failed with non-retryable error ({category}): {exc}"
                    )
                    last_error = OCRNonRetryableError(
                        f"OCR API failure on page {page_number}: {exc}",
                        category=category,
                    )
                    break

                last_error = OCRServiceError(
                    f"OCR API failure on page {page_number}: {exc}",
                    category=category,
                )
                logger.warning(
                    f"Page {page_number}: Attempt {attempt} failed with retryable error ({category}): {exc}"
                )
                if attempt == attempts:
                    break

            backoff = settings.backoff_seconds * (2 ** (attempt - 1))
            logger.info(f"Page {page_number}: Retrying in {backoff} seconds...")
            await asyncio.sleep(backoff)

        # Fallback to configured fallback provider if available and primary failed
        if self._fallback_provider:
            logger.info(f"Page {page_number}: Primary OCR failed ({last_error}). Trying fallback provider...")
            try:
                fallback_started_at = perf_counter()
                markdown = await self._fallback_provider.extract_markdown(page_number, image_path)
                logger.info(f"Page {page_number}: OCR fallback succeeded!")
                return OCRRequestResult(
                    markdown=markdown,
                    gemini_request_seconds=round(perf_counter() - fallback_started_at, 6),
                    retry_count=0,
                )
            except Exception as fallback_exc:
                logger.error(f"Page {page_number}: OCR fallback failed: {fallback_exc}")

        if last_error is not None:
            raise last_error
        raise OCRServiceError(f"OCR failed on page {page_number}.")


def classify_ocr_exception(exc: Exception) -> str:
    if hasattr(exc, "category"):
        return exc.category
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    message = str(exc).lower()

    if status_code in {401, 403} or "api key" in message or "unauth" in message or "unauthorized" in message:
        return "authentication_failure"
    if status_code == 429 or "rate limit" in message or "quota" in message:
        return "rate_limit"
    if status_code in {400, 422} or "invalid argument" in message:
        return "validation_error"
    if "timeout" in message or "timed out" in message:
        return "gemini_api_timeout"
    return "unknown_exception"


