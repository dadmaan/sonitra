import asyncio

import pytest

from sonitra.api.models import JobStatus
from sonitra.api.worker import run_render_worker


@pytest.mark.anyio
async def test_worker_sets_status_running(tmp_path, job_store, midi_fixture):
    job_id = job_store.create(midi_dir=str(midi_fixture("test_c4.mid").parent), out_dir=str(tmp_path))
    task = asyncio.create_task(run_render_worker(job_id, job_store))
    await asyncio.sleep(0.05)
    assert job_store.get(job_id).status in {JobStatus.RUNNING, JobStatus.DONE}
    await task


@pytest.mark.anyio
async def test_worker_sets_status_done_on_completion(tmp_path, job_store, midi_fixture):
    job_id = job_store.create(midi_dir=str(midi_fixture("test_c4.mid").parent), out_dir=str(tmp_path))
    await run_render_worker(job_id, job_store)
    assert job_store.get(job_id).status == JobStatus.DONE


@pytest.mark.anyio
async def test_worker_updates_succeeded_count(tmp_path, job_store, midi_fixture):
    job_id = job_store.create(midi_dir=str(midi_fixture("test_c4.mid").parent), out_dir=str(tmp_path))
    await run_render_worker(job_id, job_store)
    assert job_store.get(job_id).succeeded > 0


@pytest.mark.anyio
async def test_worker_sets_failed_on_crash(tmp_path, job_store, monkeypatch, midi_fixture):
    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("sonitra.api.worker.run_pipeline", explode)
    job_id = job_store.create(midi_dir=str(midi_fixture("test_c4.mid").parent), out_dir=str(tmp_path))
    await run_render_worker(job_id, job_store)
    assert job_store.get(job_id).status == JobStatus.FAILED


@pytest.mark.anyio
async def test_worker_respects_cancel_request(tmp_path, job_store, midi_fixture):
    job_id = job_store.create(midi_dir=str(midi_fixture("test_c4.mid").parent), out_dir=str(tmp_path))
    job_store.cancel(job_id)
    await run_render_worker(job_id, job_store)
    assert job_store.get(job_id).status == JobStatus.CANCELLED


@pytest.mark.anyio
async def test_post_job_triggers_background_render(client, tmp_path, midi_fixture):
    r = await client.post(
        "/jobs",
        json={
            "midi_dir": str(midi_fixture("test_c4.mid").parent),
            "out_dir": str(tmp_path),
            "plugin_path": None,
        },
    )
    job_id = r.json()["job_id"]
    for _ in range(20):
        await asyncio.sleep(0.1)
        status_r = await client.get(f"/jobs/{job_id}")
        if status_r.json()["status"] in {"done", "failed"}:
            break
    assert status_r.json()["status"] == "done"
