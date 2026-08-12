from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import pytest
from rich.progress import Progress, TextColumn
from rich.text import Text

import sonitra.terminal as terminal_module
from sonitra.benchmark.results import WorkerEvent
from sonitra.terminal import NullBenchmarkProgress, RichBenchmarkProgress, get_console


@pytest.fixture(autouse=True)
def _reset_console_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminal_module, "_console", None)


def test_get_console_pins_stdout_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console must keep writing to the real terminal even if something
    else (e.g. a serial-mode output guard) temporarily reassigns sys.stdout."""
    pinned_stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", pinned_stdout)

    console = get_console()
    assert console.file is pinned_stdout

    other_target = io.StringIO()
    with redirect_stdout(other_target):
        assert console.file is pinned_stdout


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
        task = progress._workers_progress._tasks[task_id]
        assert Text.from_markup(task.description).plain == (
            f"pid {os.getpid()} · baseline · render"
        )
        assert "× " not in task.description
        detail_id = progress._worker_detail_task_ids[os.getpid()]
        detail_task = progress._workers_progress._tasks[detail_id]
        assert detail_task.fields["file"] == ""
        assert detail_task.total == progress._total_files
        # completed stays at its pre-created-row default; the "stage" branch
        # must never pass completed=.
        assert detail_task.completed == 0


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
        header_id = progress._worker_task_ids[os.getpid()]
        detail_id = progress._worker_detail_task_ids[os.getpid()]
        detail_task = progress._workers_progress._tasks[detail_id]
        assert detail_task.fields["file"] == "song.mid"
        header_task = progress._workers_progress._tasks[header_id]
        assert Text.from_markup(header_task.description).plain == (
            f"pid {os.getpid()} · baseline · separate"
        )


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
        detail_id = progress._worker_detail_task_ids[os.getpid()]
        assert progress._workers_progress._tasks[detail_id].completed == 1

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
        assert progress._workers_progress._tasks[detail_id].completed == 1


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


def test_rich_benchmark_progress_creates_two_tasks_per_worker() -> None:
    n = 3
    progress = RichBenchmarkProgress(get_console(), n_workers=n)
    with progress:
        assert len(progress._worker_rows) == n
        assert len(progress._workers_progress._tasks) == 2 * n
        for header_id, detail_id in progress._worker_rows:
            header = progress._workers_progress._tasks[header_id]
            detail = progress._workers_progress._tasks[detail_id]
            assert header.fields["kind"] == "header"
            assert detail.fields["kind"] == "detail"


def test_rich_benchmark_progress_worker_rows_are_adjacent_in_task_order() -> None:
    """Header and detail tasks are added consecutively per worker, so the
    public ``tasks`` list is interleaved ``[h0, d0, h1, d1]`` in render order."""
    progress = RichBenchmarkProgress(get_console(), n_workers=2)
    with progress:
        h0, d0 = progress._worker_rows[0]
        h1, d1 = progress._worker_rows[1]
        assert [t.id for t in progress._workers_progress.tasks] == [h0, d0, h1, d1]


def test_blankable_column_blanks_matching_kind_and_delegates_otherwise() -> None:
    from sonitra.terminal import _BlankableColumn

    wrapped = TextColumn("{task.fields[file]}")
    progress = Progress()
    header_id = progress.add_task("h", file="HEADER", kind="header")
    detail_id = progress.add_task("d", file="DETAIL", kind="detail")
    header_task = progress._tasks[header_id]
    detail_task = progress._tasks[detail_id]

    header_blank = _BlankableColumn(wrapped, blank_for_kind="header")
    detail_blank = _BlankableColumn(wrapped, blank_for_kind="detail")

    assert header_blank.render(header_task).plain == ""
    assert header_blank.render(detail_task).plain == "DETAIL"
    assert detail_blank.render(header_task).plain == "HEADER"
    assert detail_blank.render(detail_task).plain == ""


def test_describe_worker_colors_fields_and_escapes_bracket_characters() -> None:
    from sonitra.terminal import (
        _STYLE_CONDITION,
        _STYLE_DEVICE,
        _STYLE_PID,
        _STYLE_STAGE,
        _STYLE_TRANSCRIBER,
    )

    progress = RichBenchmarkProgress(
        get_console(), n_workers=1, devices={"basic_pitch": "cpu"}
    )
    event = WorkerEvent(
        worker_id=1001,
        condition="reverb[wet=0.5]",
        transcriber="basic_pitch",
        midi_path="",
        status="stage",
        stage="transcribe",
        ok=True,
    )
    description = progress._describe_worker(event)
    text = Text.from_markup(description)  # must not raise MarkupError
    assert text.plain == (
        "pid 1001 · reverb[wet=0.5] × basic_pitch · transcribe · cpu"
    )

    def span_style(substring: str) -> str | None:
        for span in text.spans:
            if text.plain[span.start:span.end] == substring:
                return span.style
        return None

    assert span_style("1001") == _STYLE_PID
    assert span_style("reverb[wet=0.5]") == _STYLE_CONDITION
    assert span_style("basic_pitch") == _STYLE_TRANSCRIBER
    assert span_style("transcribe") == _STYLE_STAGE
    assert span_style("cpu") == _STYLE_DEVICE


def test_worker_event_file_field_survives_bracket_characters_in_midi_path() -> None:
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    midi_path = "my [demo] files/song.mid"
    with progress:
        progress.on_condition_started("baseline", {}, 1, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="basic_pitch",
                midi_path=midi_path,
                status="start",
                stage="transcribe",
                ok=True,
            )
        )
        detail_id = progress._worker_detail_task_ids[os.getpid()]
        file_field = progress._workers_progress._tasks[detail_id].fields["file"]
        # The stored (escaped) value parses cleanly back to the original path.
        assert Text.from_markup(file_field).plain == midi_path


def test_workers_progress_columns_blank_per_row_kind() -> None:
    from sonitra.terminal import _BlankableColumn

    progress = RichBenchmarkProgress(get_console(), n_workers=2)
    with progress:
        # Give the first worker's detail row real content so a non-blanked
        # detail column (file/text) renders non-empty instead of an
        # accidentally-empty string.
        progress.on_condition_started("baseline", {}, 3, ["basic_pitch"])
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
        header_id, detail_id = progress._worker_rows[0]
        header_task = progress._workers_progress._tasks[header_id]
        detail_task = progress._workers_progress._tasks[detail_id]
        assert detail_task.fields["file"] != ""

        def is_blank(renderable) -> bool:
            return isinstance(renderable, Text) and renderable.plain == ""

        for column in progress._workers_progress.columns:
            if isinstance(column, _BlankableColumn):
                header_blank = is_blank(column.render(header_task))
                assert header_blank == (column.blank_for_kind == "header")
                detail_blank = is_blank(column.render(detail_task))
                assert detail_blank == (column.blank_for_kind == "detail")
