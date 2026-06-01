import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from midi_renderer.api.job_store import JobStore
from midi_renderer.api.models import JobStatus

router = APIRouter()


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


@router.get("/status/{job_id}/stream")
async def stream_status(job_id: str, store: JobStore = Depends(get_job_store)) -> StreamingResponse:
    try:
        store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc

    async def event_stream():
        while True:
            record = store.get(job_id)
            payload = {
                "job_id": record.job_id,
                "status": record.status.value,
                "total": record.total,
                "succeeded": record.succeeded,
                "failed": record.failed,
                "skipped": record.skipped,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if record.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}:
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
