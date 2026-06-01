import yaml
import pytest


@pytest.mark.anyio
async def test_get_config_returns_current_config(client):
    r = await client.get("/config")
    assert r.status_code == 200
    assert "pipeline" in r.json()


@pytest.mark.anyio
async def test_get_config_rendering_mode_present(client):
    r = await client.get("/config")
    assert "rendering_mode" in r.json()["pipeline"]


@pytest.mark.anyio
async def test_put_config_updates_rendering_mode(client, config_fixture):
    with open(config_fixture("config_pedalboard_only.yaml")) as f:
        new_cfg = yaml.safe_load(f)
    r = await client.put("/config", json=new_cfg)
    assert r.status_code == 200
    get_r = await client.get("/config")
    assert get_r.json()["pipeline"]["rendering_mode"] == "pedalboard_only"


@pytest.mark.anyio
async def test_put_invalid_config_returns_422(client):
    r = await client.put("/config", json={"pipeline": {"rendering_mode": "invalid"}})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_post_job_uses_current_config(client, tmp_path, config_fixture):
    with open(config_fixture("config_pedalboard_only.yaml")) as f:
        new_cfg = yaml.safe_load(f)
    await client.put("/config", json=new_cfg)
    r = await client.post(
        "/jobs",
        json={
            "midi_dir": str(tmp_path),
            "out_dir": str(tmp_path),
        },
    )
    job_id = r.json()["job_id"]
    status_r = await client.get(f"/jobs/{job_id}")
    assert "rendering_mode" in status_r.json()
