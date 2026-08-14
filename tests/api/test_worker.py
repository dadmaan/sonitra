import asyncio

import pytest

from sonitra.api.models import JobStatus
from sonitra.api.worker import run_render_worker
from sonitra.config import PipelineConfig
from sonitra.pipeline import run_pipeline
from sonitra.storage import write_wav


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


@pytest.mark.anyio
async def test_worker_does_not_bleed_dawdreamer_state(
    tmp_path,
    job_store,
    midi_fixture,
):
    """Regression test for DawDreamer thread state bleed after API worker."""
    from sonitra.engine import RendererEngine

    job_id = job_store.create(
        midi_dir=str(midi_fixture("test_c4.mid").parent),
        out_dir=str(tmp_path / "worker_out"),
    )
    await run_render_worker(job_id, job_store)
    assert job_store.get(job_id).status == JobStatus.DONE

    # Rendering with a fresh engine after the worker must not deadlock.
    fresh_engine = RendererEngine(sample_rate=44100, block_size=512)
    result = run_pipeline(
        [midi_fixture("test_c4.mid")],
        out_dir=str(tmp_path / "fresh_out"),
        engine=fresh_engine,
    )
    assert result.succeeded == 1


def _audio_worker_config(tmp_path) -> PipelineConfig:
    """Audio-mode config mirroring ``tests/test_pipeline_audio.py``'s
    ``_audio_config`` helper: fluidsynth backend (unused/unvalidated in
    audio mode), permissive quality gates."""
    return PipelineConfig.model_validate(
        {
            "render_pipeline": {
                "synth_backend": "fluidsynth",
                "effects_chain": "none",
                "input_type": "audio",
                "bpm": 120,
                "sample_rate": 44100,
                "bit_depth": 24,
                "channels": 2,
                "duration_padding_sec": 2.0,
                "overwrite": True,
                "resume": True,
                "max_workers": 1,
                "log_level": "INFO",
            },
            "io": {
                "corpus_root": str(tmp_path),
                "output_format": "wav",
                "mp3_bitrate_kbps": 192,
                "file_naming": "{stem}",
            },
            "quality_gates": {
                "silence_threshold_rms": 0.001,
                "min_duration_sec": 0.05,
                "max_duration_deviation_sec": 5.0,
                "clip_threshold": 0.999,
            },
        }
    )


@pytest.mark.anyio
async def test_worker_audio_mode_globs_audio_files(tmp_path, job_store, dummy_audio):
    """In audio mode the worker must glob source recordings (.wav/.flac/.mp3),
    not ``*.mid``, and run the pipeline's audio path against them."""
    audio_dir = tmp_path / "audio_in"
    audio_dir.mkdir()
    write_wav(dummy_audio, audio_dir / "a.wav", sample_rate=44100, normalize=False)
    write_wav(dummy_audio, audio_dir / "b.wav", sample_rate=44100, normalize=False)
    # A stray non-audio file must be ignored in audio mode.
    (audio_dir / "ignored.mid").write_bytes(b"")

    cfg = _audio_worker_config(tmp_path)
    job_id = job_store.create(midi_dir=str(audio_dir), out_dir=str(tmp_path / "out"))

    await run_render_worker(job_id, job_store, config=cfg)

    job = job_store.get(job_id)
    assert job.status == JobStatus.DONE
    assert job.total == 2
    assert job.succeeded == 2
    assert job.failed == 0
    assert sorted(p.name for p in (tmp_path / "out").glob("*.wav")) == ["a.wav", "b.wav"]
