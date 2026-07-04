import shutil
import tempfile
from pathlib import Path
from uuid import uuid4


class LocalTempStorage:
    """Local temp storage boundary; replace with cloud storage later."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.workdir: Path | None = None

    async def __aenter__(self) -> "LocalTempStorage":
        self.root.mkdir(parents=True, exist_ok=True)
        self.workdir = Path(tempfile.mkdtemp(prefix="job-", dir=self.root))
        (self.workdir / "uploads").mkdir()
        (self.workdir / "pages").mkdir()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.cleanup()

    async def save_upload(self, filename: str, data: bytes) -> Path:
        directory = self._require_workdir() / "uploads"
        destination = directory / f"{uuid4().hex}-{Path(filename).name}"
        destination.write_bytes(data)
        return destination

    async def reserve_page_path(self, page_number: int) -> Path:
        directory = self._require_workdir() / "pages"
        return directory / f"page-{page_number:04d}.png"

    async def cleanup(self) -> None:
        if self.workdir and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir = None

    def _require_workdir(self) -> Path:
        if self.workdir is None:
            raise RuntimeError("Temporary storage has not been initialized.")
        return self.workdir
