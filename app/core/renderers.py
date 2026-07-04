from __future__ import annotations

from typing import Protocol

from app.core.examination_model import Document


class DocumentRenderer(Protocol):
    format_name: str
    file_extension: str
    media_type: str
    production_enabled: bool

    def render(self, document: Document) -> str | bytes:
        raise NotImplementedError


class DeprecatedDocxRenderer:
    format_name = "docx"
    file_extension = "docx"
    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    production_enabled = False

    def render(self, document: Document) -> bytes:
        raise RuntimeError(
            "DOCX rendering is deprecated and unavailable in the production workflow. "
            "Use the HTML renderer for production exports."
        )
