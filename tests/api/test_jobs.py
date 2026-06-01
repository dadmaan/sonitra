import pytest


@pytest.mark.anyio
async def test_post_job_returns_201(client, tmp_path):
    r = await client.post(
        "/jobs",
        json={"midi_dir": str(tmp_path), "out_dir": str(tmp_path), "plugin_path": None},
    )
    assert r.status_code == 201


@pytest.mark.anyio
async def test_post_job_returns_job_id(client, tmp_path):
    r = await client.post(
        "/jobs",
        json={"midi_dir": str(tmp_path), "out_dir": str(tmp_path), "plugin_path": None},
    )
    assert "job_id" in r.json()


@pytest.mark.anyio
async def test_post_job_status_is_pending(client, tmp_path):
    r = await client.post(
        "/jobs",
        json={"midi_dir": str(tmp_path), "out_dir": str(tmp_path), "plugin_path": None},
    )
    assert r.json()["status"] == "pending"


@pytest.mark.anyio
async def test_post_job_invalid_body_returns_422(client):
    r = await client.post("/jobs", json={})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_get_jobs_empty_list(client):
    r = await client.get("/jobs")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.anyio
async def test_get_job_by_id(client, tmp_path):
    post_r = await client.post(
        "/jobs",
        json={"midi_dir": str(tmp_path), "out_dir": str(tmp_path), "plugin_path": None},
    )
    job_id = post_r.json()["job_id"]
    get_r = await client.get(f"/jobs/{job_id}")
    assert get_r.status_code == 200
    assert get_r.json()["job_id"] == job_id


@pytest.mark.anyio
async def test_get_nonexistent_job_returns_404(client):
    r = await client.get("/jobs/ghost-id")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_delete_job_returns_204(client, tmp_path):
    post_r = await client.post(
        "/jobs",
        json={"midi_dir": str(tmp_path), "out_dir": str(tmp_path), "plugin_path": None},
    )
    job_id = post_r.json()["job_id"]
    del_r = await client.delete(f"/jobs/{job_id}")
    assert del_r.status_code == 204


@pytest.mark.anyio
async def test_delete_nonexistent_job_returns_404(client):
    r = await client.delete("/jobs/ghost-id")
    assert r.status_code == 404
