from __future__ import annotations

import asyncio
import atexit
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sonitra.api.job_store import JobStore
from sonitra.api.models import JobStatus
from sonitra.config import PipelineConfig
from sonitra.engine import RendererEngine
from sonitra.pipeline import run_pipeline


def _render_sync(
    job_id: str,
    store: JobStore,
    midi_paths: list[Path],
    out_dir: str,
    plugin_path: str | None,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> tuple[JobStatus, int, int, int]:
    succeeded = 0
    failed = 0
    skipped = 0

    cfg: PipelineConfig | None = config
    if cfg is None and config_path is not None:
        from sonitra.config import load_config
        cfg = load_config(config_path)

    if cfg is not None:
        for midi_path in midi_paths:
            if store.is_cancel_requested(job_id):
                return JobStatus.CANCELLED, succeeded, failed, skipped
            result = run_pipeline([midi_path], out_dir=out_dir, config=cfg)
            succeeded += result.succeeded
            failed += result.failed
            skipped += result.skipped
            store.update(job_id, succeeded=succeeded, failed=failed, skipped=skipped)
            time.sleep(0.01)
        return JobStatus.DONE, succeeded, failed, skipped

    engine = RendererEngine(sample_rate=44100, block_size=512)
    for midi_path in midi_paths:
        if store.is_cancel_requested(job_id):
            return JobStatus.CANCELLED, succeeded, failed, skipped
        result = run_pipeline(
            [midi_path],
            out_dir=out_dir,
            engine=engine,
            plugin_path=plugin_path,
        )
        succeeded += result.succeeded
        failed += result.failed
        skipped += result.skipped
        store.update(job_id, succeeded=succeeded, failed=failed, skipped=skipped)
        time.sleep(0.01)
    return JobStatus.DONE, succeeded, failed, skipped


_EXECUTOR = ThreadPoolExecutor(max_workers=1)
atexit.register(_EXECUTOR.shutdown, wait=False)


def _render_job_sync(job_id: str, store: JobStore, config: Any = None) -> None:
    try:
        job = store.get(job_id)
    except KeyError:
        return

    if job.status == JobStatus.CANCELLED:
        return

    store.update(job_id, status=JobStatus.RUNNING)
    midi_paths = sorted(Path(job.midi_dir).glob("*.mid"))
    store.update(job_id, total=len(midi_paths))

    start = time.perf_counter()
    if not midi_paths:
        store.update(job_id, status=JobStatus.DONE, elapsed_seconds=0.0)
        return

    try:
        status, succeeded, failed, skipped = _render_sync(
            job_id,
            store,
            midi_paths,
            job.out_dir,
            job.plugin_path,
            config=config if isinstance(config, PipelineConfig) else None,
        )
    except Exception:
        elapsed = time.perf_counter() - start
        store.update(job_id, status=JobStatus.FAILED, elapsed_seconds=elapsed)
        return

    elapsed = time.perf_counter() - start
    store.update(
        job_id,
        status=status,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        elapsed_seconds=elapsed,
    )


async def run_render_worker(job_id: str, store: JobStore, config: Any = None) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_EXECUTOR, _render_job_sync, job_id, store, config)


def schedule_render_worker(job_id: str, store: JobStore, config: Any = None) -> None:
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_EXECUTOR, _render_job_sync, job_id, store, config)
