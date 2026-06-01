import pytest


@pytest.mark.anyio
async def test_health_returns_200(client):
    r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.anyio
async def test_health_body_contains_status(client):
    r = await client.get("/health")
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_ready_returns_200(client):
    r = await client.get("/ready")
    assert r.status_code == 200


@pytest.mark.anyio
async def test_unknown_route_returns_404(client):
    r = await client.get("/nonexistent")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_cors_header_present(client):
    r = await client.options("/health", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" in r.headers
