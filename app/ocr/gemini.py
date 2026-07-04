import asyncio
from pathlib import Path

from google import genai
from google.genai import types

from app.config import settings
from app.ocr.base import OCRProvider
from app.ocr.prompts import OCR_SYSTEM_PROMPT, page_prompt


class GeminiOCRProvider(OCRProvider):
    def __init__(self) -> None:
        api_key = settings.resolved_gemini_api_key
        if not api_key:
            raise RuntimeError(
                "Gemini API key is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = settings.gemini_model

    async def extract_markdown(self, page_number: int, image_path: Path) -> str:
        return await asyncio.to_thread(self._extract_markdown_sync, page_number, image_path)

    def _extract_markdown_sync(self, page_number: int, image_path: Path) -> str:
        image_bytes = image_path.read_bytes()
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_text(text=page_prompt(page_number)),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                system_instruction=OCR_SYSTEM_PROMPT,
                temperature=0,
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError(f"Gemini returned empty OCR output for page {page_number}.")
        return text.strip()
