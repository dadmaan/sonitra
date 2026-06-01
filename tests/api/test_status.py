import json

import pytest


@pytest.mark.anyio
async def test_sse_endpoint_returns_200(client, tmp_path):
    post_r = await client.post(
        "/jobs",
        json={"midi_dir": str(tmp_path), "out_dir": str(tmp_path), "plugin_path": None},
    )
    job_id = post_r.json()["job_id"]
    async with client.stream("GET", f"/status/{job_id}/stream") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]


@pytest.mark.anyio
async def test_sse_emits_progress_event(client, tmp_path, midi_fixture):
    post_r = await client.post(
        "/jobs",
        json={
            "midi_dir": str(midi_fixture("test_c4.mid").parent),
            "out_dir": str(tmp_path),
            "plugin_path": None,
        },
    )
    job_id = post_r.json()["job_id"]
    events = []
    async with client.stream("GET", f"/status/{job_id}/stream") as r:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
            if any(e.get("status") in {"done", "failed"} for e in events):
                break
    assert any(e["job_id"] == job_id for e in events)


@pytest.mark.anyio
async def test_sse_nonexistent_job_returns_404(client):
    r = await client.get("/status/ghost-id/stream")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_sse_event_contains_required_fields(client, tmp_path, midi_fixture):
    post_r = await client.post(
        "/jobs",
        json={
            "midi_dir": str(midi_fixture("test_c4.mid").parent),
            "out_dir": str(tmp_path),
            "plugin_path": None,
        },
    )
    job_id = post_r.json()["job_id"]
    async with client.stream("GET", f"/status/{job_id}/stream") as r:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                event = json.loads(line[5:])
                assert {"job_id", "status", "succeeded", "failed", "total"}.issubset(event.keys())
                break
