from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from midi_renderer.api.job_store import JobStore
from midi_renderer.api.routers import config as config_router
from midi_renderer.api.routers import health, jobs, status
from midi_renderer.config import default_config_path, load_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.job_store = JobStore()
    app.state.config = load_config(default_config_path())
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.job_store = JobStore()
    app.state.worker_futures = set()
    app.state.config = load_config(default_config_path())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(config_router.router)
    app.include_router(status.router)
    return app
