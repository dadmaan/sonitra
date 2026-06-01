from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from sonitra.api.models import JobStatus


@dataclass
class JobRecord:
    job_id: str
    midi_dir: str
    out_dir: str
    plugin_path: str | None
    rendering_mode: str
    status: JobStatus
    total: int
    succeeded: int
    failed: int
    skipped: int
    elapsed_seconds: float
    created_at: str


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._cancel_requested: dict[str, bool] = {}
        self._lock = Lock()

    def create(self, *, midi_dir: str, out_dir: str, rendering_mode: str = "", plugin_path: str | None = None) -> str:
        job_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = JobRecord(
            job_id=job_id,
            midi_dir=midi_dir,
            out_dir=out_dir,
            plugin_path=plugin_path,
            rendering_mode=rendering_mode,
            status=JobStatus.PENDING,
            total=0,
            succeeded=0,
            failed=0,
            skipped=0,
            elapsed_seconds=0.0,
            created_at=created_at,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._cancel_requested[job_id] = False
        return job_id

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def list(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **updates) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            for field, value in updates.items():
                if hasattr(record, field):
                    setattr(record, field, value)
            return record

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            if record.status == JobStatus.PENDING:
                record.status = JobStatus.CANCELLED
            else:
                self._cancel_requested[job_id] = True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._cancel_requested.get(job_id, False)
