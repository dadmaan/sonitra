from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status

from sonitra.api.job_store import JobRecord, JobStore
from sonitra.api.models import JobRequest, JobResponse
from sonitra.api.worker import run_render_worker

router = APIRouter()


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def to_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        synth_backend=record.synth_backend,
        total=record.total,
        succeeded=record.succeeded,
        failed=record.failed,
        skipped=record.skipped,
        elapsed_seconds=record.elapsed_seconds,
        created_at=record.created_at,
    )


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    payload: JobRequest,
    request: Request,
    store: JobStore = Depends(get_job_store),
) -> JobResponse:
    synth_backend = request.app.state.config.render_pipeline.synth_backend.value
    job_id = store.create(
        midi_dir=str(payload.midi_dir),
        out_dir=str(payload.out_dir),
        synth_backend=synth_backend,
        plugin_path=str(payload.plugin_path) if payload.plugin_path else None,
    )
    task = asyncio.create_task(
        run_render_worker(job_id, store, config=request.app.state.config)
    )
    request.app.state.worker_futures.add(task)
    task.add_done_callback(request.app.state.worker_futures.discard)
    return to_response(store.get(job_id))


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(store: JobStore = Depends(get_job_store)) -> list[JobResponse]:
    return [to_response(record) for record in store.list()]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobResponse:
    try:
        return to_response(store.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str, store: JobStore = Depends(get_job_store)) -> None:
    try:
        store.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc
