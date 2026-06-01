import pytest

from midi_renderer.api.job_store import JobStore
from midi_renderer.api.models import JobStatus


def test_create_job_returns_unique_id():
    store = JobStore()
    id1 = store.create(midi_dir="/a", out_dir="/b")
    id2 = store.create(midi_dir="/a", out_dir="/b")
    assert id1 != id2


def test_job_initially_pending():
    store = JobStore()
    job_id = store.create(midi_dir="/a", out_dir="/b")
    assert store.get(job_id).status == JobStatus.PENDING


def test_update_job_status():
    store = JobStore()
    job_id = store.create(midi_dir="/a", out_dir="/b")
    store.update(job_id, status=JobStatus.RUNNING, succeeded=5)
    assert store.get(job_id).status == JobStatus.RUNNING
    assert store.get(job_id).succeeded == 5


def test_get_nonexistent_job_raises():
    store = JobStore()
    with pytest.raises(KeyError):
        store.get("ghost-id")


def test_list_returns_all_jobs():
    store = JobStore()
    store.create(midi_dir="/a", out_dir="/b")
    store.create(midi_dir="/c", out_dir="/d")
    assert len(store.list()) == 2


def test_cancel_pending_job():
    store = JobStore()
    job_id = store.create(midi_dir="/a", out_dir="/b")
    store.cancel(job_id)
    assert store.get(job_id).status == JobStatus.CANCELLED


def test_cancel_running_job_sets_flag():
    store = JobStore()
    job_id = store.create(midi_dir="/a", out_dir="/b")
    store.update(job_id, status=JobStatus.RUNNING)
    store.cancel(job_id)
    assert store.is_cancel_requested(job_id)
