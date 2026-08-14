from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from sonitra.api.job_store import JobStore
from sonitra.api.models import JobStatus
from sonitra.config import InputType, PipelineConfig
from sonitra.corpus import discover_audio_files
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


_DAWDREAMER_LOCK = asyncio.Lock()


def _render_job_sync(job_id: str, store: JobStore, config: Any = None) -> None:
    try:
        job = store.get(job_id)
    except KeyError:
        return

    if job.status == JobStatus.CANCELLED:
        return

    store.update(job_id, status=JobStatus.RUNNING)
    cfg = config if isinstance(config, PipelineConfig) else None
    if cfg is not None and cfg.render_pipeline.input_type == InputType.AUDIO:
        midi_paths = discover_audio_files(Path(job.midi_dir))
    else:
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


async def run_render_worker(
    job_id: str,
    store: JobStore,
    config: Any = None,
) -> None:
    """Run a render job while holding the DawDreamer serialization lock.

    DawDreamer/JUCE global state in this environment is corrupted when Faust
    runs in a subprocess or when the event loop yields during rendering. All
    rendering therefore runs synchronously on the caller's event loop while
    holding the lock. A short, lock-free yield at entry lets a PENDING
    cancellation request land before rendering starts.
    """
    async with _DAWDREAMER_LOCK:
        try:
            job = store.get(job_id)
        except KeyError:
            return
        if job.status == JobStatus.CANCELLED:
            return

    # Release the lock and yield so a concurrently dispatched DELETE can set
    # the job to CANCELLED before we start the synchronous render.
    await asyncio.sleep(0.01)

    async with _DAWDREAMER_LOCK:
        try:
            job = store.get(job_id)
        except KeyError:
            return
        if job.status == JobStatus.CANCELLED:
            return
        if store.is_cancel_requested(job_id):
            store.update(job_id, status=JobStatus.CANCELLED)
            return

        _render_job_sync(job_id, store, config)
