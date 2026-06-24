import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from sonitra.api.app import create_app
from sonitra.api.job_store import JobStore
import shutil


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture
async def client(app):
    from sonitra.config import default_config_path, load_config

    app.state.job_store = JobStore()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    if app.state.worker_futures:
        await asyncio.gather(*list(app.state.worker_futures), return_exceptions=True)
        app.state.worker_futures.clear()
    # Reset config so tests that PUT a different config do not leak state.
    app.state.config = load_config(default_config_path())


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
