from __future__ import annotations

import os

import pytest

import sonitra.terminal as terminal_module
from sonitra.benchmark.results import WorkerEvent
from sonitra.terminal import NullBenchmarkProgress, RichBenchmarkProgress, get_console


@pytest.fixture(autouse=True)
def _reset_console_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminal_module, "_console", None)


def test_rich_benchmark_progress_live_is_transient() -> None:
    """The outer Live must not leave a stale frame printed after it exits."""
    console = get_console()
    progress = RichBenchmarkProgress(console, n_workers=1)
    assert progress._live.transient is True


def test_rich_benchmark_progress_stage_render_updates_description_and_clears_file() -> None:
    """A ``stage``/``render`` event (no preceding ``start``) assigns the row,
    shows a stage chip with no dangling ``× `` (empty transcriber), clears any
    stale filename, and shows the row's M/N total during render too."""
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    with progress:
        progress.on_condition_started("baseline", {}, 3, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="",
                midi_path="",
                status="stage",
                stage="render",
                ok=True,
            )
        )
        task_id = progress._worker_task_ids[os.getpid()]
        task = progress._workers_progress.tasks[task_id]
        assert task.description == f"pid {os.getpid()} · baseline · render"
        assert "× " not in task.description
        assert task.fields["file"] == ""
        assert task.total == progress._total_files
        # completed stays at its pre-created-row default; the "stage" branch
        # must never pass completed=.
        assert task.completed == 0


def test_rich_benchmark_progress_stage_separate_sets_file_to_midi_path() -> None:
    """A ``stage``/``separate`` event updates ``file`` to the real, known
    filename (fixes the stale-filename display bug the "description only"
    rule would otherwise have left in place)."""
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    with progress:
        progress.on_condition_started("baseline", {}, 2, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="",
                midi_path="song.mid",
                status="stage",
                stage="separate",
                ok=True,
            )
        )
        task_id = progress._worker_task_ids[os.getpid()]
        task = progress._workers_progress.tasks[task_id]
        assert task.fields["file"] == "song.mid"
        assert task.description == f"pid {os.getpid()} · baseline · separate"


def test_rich_benchmark_progress_stage_event_does_not_reset_completed_counter() -> None:
    """Narrow regression lock: a ``stage`` event fired between a file's
    ``done`` and the next file's ``start`` must not roll back the row's
    ``completed`` counter. This does NOT claim the M/N counter is correct
    overall -- the pre-existing ``completed=0``-on-every-``"start"`` bug is
    untouched and remains deferred."""
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    with progress:
        progress.on_condition_started("baseline", {}, 2, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="basic_pitch",
                midi_path="song1.mid",
                status="start",
                stage="transcribe",
                ok=True,
            )
        )
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="basic_pitch",
                midi_path="song1.mid",
                status="done",
                ok=True,
            )
        )
        task_id = progress._worker_task_ids[os.getpid()]
        assert progress._workers_progress.tasks[task_id].completed == 1

        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="",
                midi_path="song2.mid",
                status="stage",
                stage="separate",
                ok=True,
            )
        )
        assert progress._workers_progress.tasks[task_id].completed == 1


def test_rich_benchmark_progress_stage_event_assigns_row_before_start() -> None:
    """A ``stage`` event as the very first event from a pid assigns it a row
    and adds its pid to the header, with no ``"start"`` yet."""
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    with progress:
        progress.on_condition_started("baseline", {}, 1, ["basic_pitch"])
        assert os.getpid() not in progress._worker_task_ids

        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="",
                midi_path="",
                status="stage",
                stage="render",
                ok=True,
            )
        )
        assert os.getpid() in progress._worker_task_ids
        assert os.getpid() in progress._pids


def test_null_benchmark_progress_worker_event_stage_is_noop() -> None:
    progress = NullBenchmarkProgress()
    assert (
        progress.on_worker_event(
            WorkerEvent(
                worker_id=1,
                condition="baseline",
                transcriber="",
                midi_path="",
                status="stage",
                stage="render",
                ok=True,
            )
        )
        is None
    )
