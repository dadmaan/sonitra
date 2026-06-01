import json

import pytest
from pydantic import ValidationError

from sonitra.api.models import JobRequest, JobResponse, JobStatus


def test_job_request_requires_midi_dir():
    with pytest.raises(ValidationError):
        JobRequest(out_dir="/tmp/out", plugin_path="/vst/synth.vst3")


def test_job_request_rejects_nonexistent_path(tmp_path):
    with pytest.raises(ValidationError):
        JobRequest(midi_dir="/does/not/exist", out_dir=str(tmp_path), plugin_path="/vst/synth.vst3")


def test_job_request_accepts_valid_paths(tmp_path):
    req = JobRequest(midi_dir=str(tmp_path), out_dir=str(tmp_path), plugin_path=None)
    assert req.midi_dir == tmp_path


def test_job_status_enum_values():
    assert JobStatus.PENDING != JobStatus.RUNNING
    assert JobStatus.DONE != JobStatus.FAILED


def test_job_response_serialises_to_json():
    resp = JobResponse(
        job_id="abc123",
        status=JobStatus.PENDING,
        total=10,
        succeeded=0,
        failed=0,
        skipped=0,
        elapsed_seconds=0.0,
        created_at="2026-05-29T08:00:00Z",
    )
    json.dumps(resp.model_dump())


def test_job_response_progress_field():
    resp = JobResponse(
        job_id="x",
        status=JobStatus.RUNNING,
        total=10,
        succeeded=3,
        failed=0,
        skipped=0,
        elapsed_seconds=5.0,
        created_at="2026-05-29T08:00:00Z",
    )
    assert resp.progress_pct == 30.0
