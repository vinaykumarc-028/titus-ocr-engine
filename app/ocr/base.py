from abc import ABC, abstractmethod
from pathlib import Path


class OCRProvider(ABC):
    @abstractmethod
    async def extract_markdown(self, page_number: int, image_path: Path) -> str:
        raise NotImplementedError
