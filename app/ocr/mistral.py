import base64
import httpx
from pathlib import Path

from app.config import settings
from app.ocr.base import OCRProvider
from app.ocr.prompts import OCR_SYSTEM_PROMPT, page_prompt


class MistralOCRProvider(OCRProvider):
    def __init__(self) -> None:
        self.api_key = settings.mistral_api_key
        if not self.api_key:
            raise RuntimeError(
                "Mistral API key is not configured. Set MISTRAL_API_KEY."
            )
        self.model = settings.mistral_model

    async def extract_markdown(self, page_number: int, image_path: Path) -> str:
        # Convert image to base64
        image_bytes = image_path.read_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if "ocr" in self.model.lower():
            # Specialized Mistral OCR API (/v1/ocr)
            payload = {
                "model": self.model,
                "document": {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}"
                }
            }
            async with httpx.AsyncClient(timeout=settings.ocr_timeout_seconds) as client:
                response = await client.post(
                    "https://api.mistral.ai/v1/ocr",
                    headers=headers,
                    json=payload,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Mistral OCR API returned error {response.status_code}: {response.text}"
                    )

                data = response.json()
                try:
                    text = data["pages"][0]["markdown"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected response format from Mistral OCR: {data}")

                if not text:
                    raise RuntimeError(
                        f"Mistral OCR returned empty output for page {page_number}."
                    )

                return text.strip()
        else:
            # Mistral Chat Completions API with vision models (e.g. pixtral)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": OCR_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": page_prompt(page_number),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0,
            }

            async with httpx.AsyncClient(timeout=settings.ocr_timeout_seconds) as client:
                response = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Mistral API returned error {response.status_code}: {response.text}"
                    )

                data = response.json()
                try:
                    text = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected response format from Mistral: {data}")

                if not text:
                    raise RuntimeError(
                        f"Mistral returned empty OCR output for page {page_number}."
                    )

                return text.strip()
