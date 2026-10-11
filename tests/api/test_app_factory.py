"""Tests for the FastAPI app factory's optional config injection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sonitra.api.app import create_app
from sonitra.config import load_config


def _build_app(config_fixture: Callable[[str], Path]) -> FastAPI:
    return create_app(config=load_config(config_fixture("config_api.yaml")))


@pytest.mark.anyio
async def test_create_app_serves_the_injected_config(config_fixture) -> None:
    app = _build_app(config_fixture)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/config")

    assert r.status_code == 200
    body = r.json()
    assert body["render_pipeline"]["sample_rate"] == 48000
    assert body["io"]["file_naming"] == "{stem}-api"


@pytest.mark.anyio
async def test_lifespan_keeps_the_injected_config(config_fixture) -> None:
    cfg = load_config(str(config_fixture("config_api.yaml")))
    app = create_app(config=cfg)

    # Startup must not overwrite a config the caller supplied: the app has to
    # keep serving the same values before and after the lifespan runs.
    assert app.state.config is cfg
    async with app.router.lifespan_context(app):
        assert app.state.config is cfg