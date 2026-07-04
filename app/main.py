import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.api.dev_validation import router as dev_validation_router
from app.api.upload import router as upload_router
from app.api.jobs import router as jobs_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Project TITUS-082 Ingestion OCR Pipeline",
    version="0.1.0",
)

# Include Routers
app.include_router(upload_router)
app.include_router(dev_validation_router)
app.include_router(jobs_router)

# Mount local uploads/pages directory to serve page images
app.mount("/static/titus-082", StaticFiles(directory=".tmp/titus-082"), name="titus-static")

# Mount React static assets directory if compiled
assets_path = Path("TITUS-Document-Intelligence/dist/assets")
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="react-assets")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/{catchall:path}")
async def serve_react_app(catchall: str):
    index_file = Path("TITUS-Document-Intelligence/dist/index.html")
    if index_file.exists():
        return FileResponse(index_file)
    
    # Fallback to dev validation page if frontend is not built
    from app.api.dev_validation import _VALIDATION_HTML
    return HTMLResponse(_VALIDATION_HTML)

