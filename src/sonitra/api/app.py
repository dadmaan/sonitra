from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sonitra.api.job_store import JobStore
from sonitra.api.routers import config as config_router
from sonitra.api.routers import health, jobs, status
from sonitra.config import PipelineConfig, default_config_path, load_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup resets the job store only: a config supplied by the caller must
    # survive startup untouched.
    app.state.job_store = JobStore()
    yield


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    """Build the app; ``config`` overrides the shipped default pipeline config."""
    app = FastAPI(lifespan=lifespan)
    app.state.job_store = JobStore()
    app.state.worker_futures = set()
    app.state.config = config if config is not None else load_config(default_config_path())
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
