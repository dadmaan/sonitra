from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import pytest
from rich.console import Console
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
    stale filename, and shows the row's M/N total during render too.

    ``render`` fires exactly once per condition, before any per-file events,
    so it is also the row's condition-boundary marker: it resets ``completed``
    to 0 for the new condition (see
    ``test_rich_benchmark_progress_render_stage_resets_completed_for_new_condition``).
    No other ``stage`` kind (e.g. ``separate``) may touch ``completed``."""
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
    """Narrow regression lock: a ``stage``/``separate`` event fired between a
    file's ``done`` and the next file's ``start`` must not roll back the
    row's ``completed`` counter. Only a ``stage``/``render`` event (a new
    condition boundary) may reset it -- see
    ``test_rich_benchmark_progress_completed_accumulates_across_files_in_a_condition``."""
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


def test_rich_benchmark_progress_completed_accumulates_across_files_in_a_condition() -> None:
    """Regression test for the completed=0-on-every-"start" bug: completed
    must accumulate 0 -> 1 -> 2 -> 3 across successive files within one
    condition, not reset back to 0 on each file's "start" event."""
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
        detail_id = progress._worker_detail_task_ids[os.getpid()]
        assert progress._workers_progress._tasks[detail_id].completed == 0

        for i in range(1, 4):
            progress.on_worker_event(
                WorkerEvent(
                    worker_id=os.getpid(),
                    condition="baseline",
                    transcriber="basic_pitch",
                    midi_path=f"song{i}.mid",
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
                    midi_path=f"song{i}.mid",
                    status="done",
                    ok=True,
                )
            )
            assert progress._workers_progress._tasks[detail_id].completed == i


def test_rich_benchmark_progress_render_stage_resets_completed_for_new_condition() -> None:
    """A worker (process-pool pid) reused for a second condition must have its
    row's completed counter reset back to 0 by that condition's "render"
    stage event -- otherwise progress from the previous condition would leak
    into the new one's M/N display."""
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    with progress:
        progress.on_condition_started("baseline", {}, 1, ["basic_pitch"])
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

        progress.on_condition_started("no_reverb", {}, 1, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="no_reverb",
                transcriber="",
                midi_path="",
                status="stage",
                stage="render",
                ok=True,
            )
        )
        assert progress._workers_progress._tasks[detail_id].completed == 0


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


def test_format_file_field_empty_path_returns_empty_string() -> None:
    from sonitra.terminal import _format_file_field

    assert _format_file_field("") == ""


def test_format_file_field_shows_parent_dir_and_basename() -> None:
    """Only the immediate parent directory is kept -- the rest of a corpus
    path (dataset root, nested year/composer dirs, etc.) is boilerplate that
    doesn't help identify the file in a live display."""
    from sonitra.terminal import _format_file_field

    assert (
        _format_file_field("/corpus/maestro-v3/midi/2018/song.mid")
        == "2018/song.mid"
    )


def test_format_file_field_drops_parent_when_path_has_no_directory() -> None:
    from sonitra.terminal import _format_file_field

    assert _format_file_field("song.mid") == "song.mid"


def test_format_file_field_short_path_under_max_len_is_unchanged() -> None:
    from sonitra.terminal import _format_file_field

    assert _format_file_field("2018/song.mid", max_len=40) == "2018/song.mid"


def test_format_file_field_middle_truncates_long_names_keeping_head_and_tail() -> None:
    """Long filenames (e.g. maestro-v3's) are truncated in the middle, not the
    end, so both the start and a distinguishing suffix (often an index like
    "--4.midi") stay visible instead of being cut off."""
    from sonitra.terminal import _format_file_field

    long_path = (
        "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--4.midi"
    )
    result = _format_file_field(long_path, max_len=20)
    assert len(result) == 20
    assert "…" in result
    assert long_path.startswith(result.split("…")[0])
    assert long_path.endswith(result.split("…")[1])


def test_worker_event_file_field_is_truncated_for_long_paths() -> None:
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    long_path = "/corpus/maestro-v3/midi/2018/" + ("x" * 100) + ".mid"
    with progress:
        progress.on_condition_started("baseline", {}, 1, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="baseline",
                transcriber="basic_pitch",
                midi_path=long_path,
                status="start",
                stage="transcribe",
                ok=True,
            )
        )
        detail_id = progress._worker_detail_task_ids[os.getpid()]
        file_field = progress._workers_progress._tasks[detail_id].fields["file"]
        plain = Text.from_markup(file_field).plain
        assert len(plain) <= 40
        assert "…" in plain


def test_worker_detail_row_file_immediately_follows_arrow_no_gap() -> None:
    """Regression: the filename must render right after the "->" arrow, not
    padded out to the width of the (much longer) header description column.

    Header and detail rows previously shared one un-blanked description
    column ("{task.description}"): its width is the max over every row, so a
    long header description (pid/condition/transcriber/stage/device) forced
    the detail row's short "->" to be right-padded out to that width before
    the separate ``file`` column even started, leaving a wall of blank space
    between the arrow and the filename.
    """
    output = io.StringIO()
    console = Console(file=output, width=200, force_terminal=True, color_system=None)
    progress = RichBenchmarkProgress(console, n_workers=1)
    with progress:
        progress.on_condition_started("baseline", {}, 1, ["basic_pitch"])
        progress.on_worker_event(
            WorkerEvent(
                worker_id=os.getpid(),
                condition="a-fairly-long-condition-name-here",
                transcriber="basic_pitch",
                midi_path="song.mid",
                status="start",
                stage="transcribe",
                ok=True,
            )
        )
        console.print(progress._workers_progress)
    rendered = output.getvalue()
    detail_line = next(line for line in rendered.splitlines() if "↳" in line)
    arrow_index = detail_line.index("↳")
    file_index = detail_line.index("song.mid")
    assert file_index - arrow_index <= 5


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
