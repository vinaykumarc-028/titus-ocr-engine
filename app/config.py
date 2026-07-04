from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    mistral_api_key: str | None = None
    mistral_model: str = "pixtral-12b-2409"
    primary_ocr_engine: str = "mistral"
    ocr_timeout_seconds: int = 180
    max_retries: int = 3
    backoff_seconds: float = 1.5
    temp_root: Path = Path(".tmp/titus-082")
    max_image_mb: int = 10
    max_pdf_mb: int = 50
    max_pdf_pages: int = 25
    max_image_side: int = 1800
    pdf_render_scale: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_gemini_api_key(self) -> str | None:
        return self.gemini_api_key or self.google_api_key

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_mb * 1024 * 1024

    @property
    def max_pdf_bytes(self) -> int:
        return self.max_pdf_mb * 1024 * 1024

    @property
    def max_pages_per_job(self) -> int:
        return self.max_pdf_pages


settings = Settings()
