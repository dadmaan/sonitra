import json
from pathlib import Path

import pytest


SNAPSHOT_PATH = Path(__file__).parent / "openapi_snapshot.json"


@pytest.mark.anyio
async def test_openapi_schema_is_accessible(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200


@pytest.mark.anyio
async def test_openapi_schema_contains_jobs_path(client):
    schema = (await client.get("/openapi.json")).json()
    assert "/jobs" in schema["paths"]


@pytest.mark.anyio
async def test_openapi_schema_unchanged(client):
    schema = (await client.get("/openapi.json")).json()
    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert schema == expected
