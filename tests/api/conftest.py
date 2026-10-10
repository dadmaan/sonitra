import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from sonitra.api.app import create_app
from sonitra.api.job_store import JobStore
from sonitra.config import load_config
import shutil

# Resolved from this file so the suite does not depend on the working directory.
_API_CONFIG_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "config_api.yaml"


@pytest.fixture(scope="session")
def app():
    return create_app(config=load_config(_API_CONFIG_PATH))


@pytest.fixture
async def client(app):
    app.state.job_store = JobStore()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    if app.state.worker_futures:
        await asyncio.gather(*list(app.state.worker_futures), return_exceptions=True)
        app.state.worker_futures.clear()
    # Reloaded from disk, not reused: tests PUT and mutate configs in place.
    app.state.config = load_config(_API_CONFIG_PATH)


@pytest.fixture
def job_store():
    return JobStore()


@pytest.fixture
def midi_corpus(tmp_path, midi_fixture):
    corpus_dir = tmp_path / "midi_corpus"
    corpus_dir.mkdir()
    for name in ["test_c4.mid", "test_polyphonic.mid", "test_empty.mid"]:
        shutil.copy(midi_fixture(name), corpus_dir / name)
    return corpus_dir


@pytest.fixture
def large_midi_corpus(tmp_path, midi_fixture):
    corpus_dir = tmp_path / "large_midi_corpus"
    corpus_dir.mkdir()
    src = midi_fixture("test_c4.mid")
    for idx in range(200):
        shutil.copy(src, corpus_dir / f"sample_{idx}.mid")
    return corpus_dir
