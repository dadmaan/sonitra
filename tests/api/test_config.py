from __future__ import annotations

import yaml
import pytest


@pytest.mark.anyio
async def test_get_config_returns_current_config(client):
    r = await client.get("/config")
    assert r.status_code == 200
    assert "render_pipeline" in r.json()


@pytest.mark.anyio
async def test_get_config_synth_backend_present(client):
    r = await client.get("/config")
    assert "synth_backend" in r.json()["render_pipeline"]


@pytest.mark.anyio
async def test_put_config_updates_synth_backend(client, config_fixture):
    with open(config_fixture("config_pedalboard_only.yaml")) as f:
        new_cfg = yaml.safe_load(f)
    r = await client.put("/config", json=new_cfg)
    assert r.status_code == 200
    get_r = await client.get("/config")
    assert get_r.json()["render_pipeline"]["synth_backend"] == "pedalboard_instrument"


@pytest.mark.anyio
async def test_put_invalid_config_returns_422(client):
    r = await client.put("/config", json={"render_pipeline": {"synth_backend": "invalid_backend"}})
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
    assert "synth_backend" in status_r.json()
