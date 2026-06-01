import asyncio
import shutil

import pytest


@pytest.mark.anyio
@pytest.mark.integration
async def test_full_render_job_via_api(client, tmp_path, midi_fixture):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    midi_dir = tmp_path / "corpus"
    midi_dir.mkdir()
    for name in ["test_c4.mid", "test_polyphonic.mid"]:
        shutil.copy(midi_fixture(name), midi_dir / name)
    r = await client.post(
        "/jobs",
        json={"midi_dir": str(midi_dir), "out_dir": str(out_dir), "plugin_path": None},
    )
    assert r.status_code == 201
    job_id = r.json()["job_id"]

    for _ in range(50):
        await asyncio.sleep(0.2)
        status = (await client.get(f"/jobs/{job_id}")).json()
        if status["status"] in {"done", "failed"}:
            break

    assert status["status"] == "done"
    assert status["failed"] == 0
    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) == len(list(midi_dir.glob("*.mid")))


@pytest.mark.anyio
@pytest.mark.integration
async def test_cancel_running_job_via_api(client, tmp_path, large_midi_corpus):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    r = await client.post(
        "/jobs",
        json={"midi_dir": str(large_midi_corpus), "out_dir": str(out_dir), "plugin_path": None},
    )
    job_id = r.json()["job_id"]
    await asyncio.sleep(0.3)
    del_r = await client.delete(f"/jobs/{job_id}")
    assert del_r.status_code == 204
    await asyncio.sleep(0.5)
    status = (await client.get(f"/jobs/{job_id}")).json()
    assert status["status"] == "cancelled"
