from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, computed_field, model_validator


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRequest(BaseModel):
    midi_dir: Path
    out_dir: Path
    plugin_path: Path | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> "JobRequest":
        if not self.midi_dir.exists():
            raise ValueError("midi_dir does not exist")
        if not self.out_dir.exists():
            raise ValueError("out_dir does not exist")
        if self.plugin_path is not None and not self.plugin_path.exists():
            raise ValueError("plugin_path does not exist")
        return self


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    rendering_mode: str = ""
    total: int
    succeeded: int
    failed: int
    skipped: int
    elapsed_seconds: float
    created_at: str

    @computed_field
    @property
    def progress_pct(self) -> float:
        if self.total <= 0:
            return 0.0
        return (self.succeeded / self.total) * 100.0


class JobLogEntry(BaseModel):
    midi: str
    output: str
    status: str
    error: str | None = None


class SSEEvent(BaseModel):
    job_id: str
    status: JobStatus
    total: int
    succeeded: int
    failed: int
    skipped: int
