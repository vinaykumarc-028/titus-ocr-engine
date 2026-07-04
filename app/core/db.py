import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger("ocr_pipeline")


class JSONDatabase:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(JSONDatabase, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path: Path | None = None) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.db_path = db_path or (settings.temp_root / "db.json")
        self._initialized = True
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        with self._lock:
            if not self.db_path.parent.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.db_path.exists():
                self._save_data({"jobs": {}, "audit_logs": []})

    def _load_data(self) -> dict[str, Any]:
        try:
            if self.db_path.exists():
                return json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Failed to load JSON database: {exc}")
        return {"jobs": {}, "audit_logs": []}

    def _save_data(self, data: dict[str, Any]) -> None:
        try:
            self.db_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.error(f"Failed to write JSON database: {exc}")

    def save_job(self, job_id: str, job_data: dict[str, Any]) -> None:
        with self._lock:
            data = self._load_data()
            job_data["last_modified"] = time.time()
            data["jobs"][job_id] = job_data
            self._save_data(data)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._load_data()
            return data["jobs"].get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_data()
            # Return sorted by creation time descending
            jobs = list(data["jobs"].values())
            jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
            return jobs

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            data = self._load_data()
            if job_id in data["jobs"]:
                del data["jobs"][job_id]
                self._save_data(data)
                return True
            return False

    def log_audit_event(self, action: str, job_id: str, job_name: str, operator_id: str = "Operator", details: str = "") -> None:
        with self._lock:
            data = self._load_data()
            event = {
                "timestamp": time.time(),
                "action": action,
                "job_id": job_id,
                "job_name": job_name,
                "operator_id": operator_id,
                "details": details
            }
            data["audit_logs"].append(event)
            self._save_data(data)

    def get_audit_logs(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_data()
            logs = list(data.get("audit_logs", []))
            logs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            return logs
