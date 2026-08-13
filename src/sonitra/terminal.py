from __future__ import annotations

import logging
import sys
from contextlib import ExitStack
from pathlib import PurePath
from types import TracebackType
from typing import Any, Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from sonitra.benchmark.results import BenchmarkRecord, WorkerEvent
from sonitra.config import PipelineConfig

_console: Console | None = None
logger = logging.getLogger(__name__)


def get_console(*, quiet: bool = False, no_color: bool = False) -> Console:
    """Return the process-wide rich Console singleton.

    The console is created lazily on first call; the first call wins the
    ``quiet`` and ``no_color`` flags. Later calls with different flags return
    the existing console unchanged — create a local ``Console`` if you need a
    one-off differently-configured console. TTY/colour detection is left to
    rich's defaults.

    ``file`` is pinned to ``sys.stdout`` at construction time rather than left
    to rich's default dynamic lookup, so the console keeps writing to the real
    terminal even if ``sys.stdout`` is later reassigned (e.g. by a serial-mode
    output guard redirecting a noisy backend's prints away from the display).
    """
    global _console
    if _console is None:
        _console = Console(quiet=quiet, no_color=no_color, file=sys.stdout)
    return _console


def effective_log_level(cfg: PipelineConfig) -> str:
    """Resolve the effective root log level from a config.

    Precedence: ``observability.log_level`` → ``pipeline.log_level`` →
    ``"INFO"``. The result is normalized to uppercase.
    """
    level = cfg.observability.log_level or cfg.pipeline.log_level or "INFO"
    return level.upper()


def setup_logging(level: str = "INFO", *, console: Console | None = None) -> None:
    """Configure the root logger with a rich handler.

    Idempotent: sets the ROOT logger's level and attaches a single
    ``RichHandler`` if none is attached yet. Safe to call repeatedly (e.g. from
    a CLI callback and again per-command); re-application only updates the
    level and never duplicates handlers.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(isinstance(handler, RichHandler) for handler in root.handlers):
        handler = RichHandler(
            console=console or get_console(),
            rich_tracebacks=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


def set_log_level(level: str) -> None:
    """Update only the ROOT logger level (e.g. after a config reloads)."""
    logging.getLogger().setLevel(level.upper())


def _format_file_field(midi_path: str, *, max_len: int = 40) -> str:
    """Shorten a worker's current file for the detail row's ``file`` column.

    Keeps only the immediate parent directory plus the basename (dataset
    roots and nested composer/year directories are boilerplate that doesn't
    help identify the file), then middle-truncates past *max_len* so both the
    start and a distinguishing suffix (e.g. maestro-v3's ``--4.midi`` index)
    stay visible instead of one end being cut off.
    """
    if not midi_path:
        return ""
    path = PurePath(midi_path)
    display = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
    if len(display) <= max_len:
        return display
    head_len = (max_len - 1) * 2 // 3
    tail_len = max_len - 1 - head_len
    return f"{display[:head_len]}…{display[-tail_len:]}"


class FilesPerSecondColumn(ProgressColumn):
    """Render a task's progress speed as ``N.N files/s``."""

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        return Text(
            "?" if speed is None else f"{speed:.1f} files/s",
            style="progress.data.speed",
        )


class _DescriptionColumn(ProgressColumn):
    """Render the header row's full description, or the detail row's arrow
    + filename, as a single column shared by both row kinds.

    Splitting these into two columns (one for "{task.description}", one for
    the file field) left the short detail-row text ("->", or "-> filename")
    padded out to the width of the much longer header description before the
    filename even started -- rich's ``Table`` sizes each column to its widest
    cell across *all* rows, including blanked ones. Folding both into one
    column means any such padding lands after the detail row's text (before
    the bar column), not as a gap between the arrow and the filename.
    """

    def render(self, task: Task) -> RenderableType:
        if task.fields.get("kind") == "detail":
            file_field = task.fields.get("file", "")
            markup = f"↳ [dim]{file_field}[/dim]" if file_field else "↳"
            return Text.from_markup(markup)
        return Text.from_markup(task.description)


class _BlankableColumn(ProgressColumn):
    """Wrap a ``ProgressColumn``, blanking it for one row ``kind``.

    ``Progress`` applies a single column list to every task it renders. To give
    header and detail rows different fields while still sharing one Progress,
    each row-kind-specific column blanks itself (``Text("")``) for the other
    kind. The wrapped column's sizing ``Column`` is preserved via
    ``get_table_column`` delegation so column widths still align across rows.
    """

    def __init__(self, wrapped: ProgressColumn, *, blank_for_kind: str) -> None:
        self._wrapped = wrapped
        self.blank_for_kind = blank_for_kind
        super().__init__()  # intentionally no table_column

    def render(self, task: Task) -> RenderableType:
        if task.fields.get("kind") == self.blank_for_kind:
            return Text("")
        # Calling the wrapped column (not .render) keeps its __call__ path.
        return self._wrapped(task)

    def get_table_column(self) -> Any:
        return self._wrapped.get_table_column().copy()


class BenchmarkProgress(Protocol):
    """Callbacks for live benchmark progress displays.

    Implementations are invoked from the benchmark runner (any thread) and the
    CLI. ``NullBenchmarkProgress`` is the no-op default; ``RichBenchmarkProgress``
    renders a live progress display.
    """

    def on_condition_started(
        self,
        condition_name: str,
        overrides: dict,
        total_files: int,
        transcriber_names: list[str],
    ) -> None: ...

    def on_worker_event(self, event: WorkerEvent) -> None: ...

    def on_condition_done(self, condition_name: str) -> None: ...


class NullBenchmarkProgress:
    """No-op :class:`BenchmarkProgress` implementation."""

    def on_condition_started(
        self,
        condition_name: str,
        overrides: dict,
        total_files: int,
        transcriber_names: list[str],
    ) -> None:
        pass

    def on_worker_event(self, event: WorkerEvent) -> None:
        pass

    def on_condition_done(self, condition_name: str) -> None:
        pass


_STYLE_PID = "dim"
_STYLE_CONDITION = "cyan"
_STYLE_TRANSCRIBER = "magenta"
_STYLE_STAGE = "bold yellow"
_STYLE_DEVICE = "green"


class RichBenchmarkProgress:
    """Live rich display of benchmark progress.

    Shows a header line (run-level aggregate, device chips, worker pids and any
    failures), an outer task counting every file across all conditions/
    transcribers, and one row per pool worker showing the active
    ``condition × transcriber``, the current pipeline stage (``render`` /
    ``separate`` / ``transcribe``), the current file, and per-file progress.
    Updates may arrive from any thread — rich's Progress/Live objects are
    RLock-protected. The display is started lazily on the first condition (or
    eagerly via the context manager) and is safe to use both ways.
    """

    def __init__(
        self,
        console: Console,
        *,
        n_workers: int = 1,
        devices: dict[str, str] | None = None,
    ) -> None:
        self.console = console
        self.n_workers = n_workers
        self.devices = devices or {}
        self._pids: set[int] = set()
        self.header = Text("benchmark", style="bold")
        self._failed = 0
        self._outer_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            FilesPerSecondColumn(),
            "ETA:",
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
            transient=True,
        )
        self._workers_progress = Progress(
            _BlankableColumn(SpinnerColumn(), blank_for_kind="detail"),
            _DescriptionColumn(),
            _BlankableColumn(BarColumn(bar_width=10), blank_for_kind="header"),
            _BlankableColumn(MofNCompleteColumn(), blank_for_kind="header"),
            _BlankableColumn(TimeElapsedColumn(), blank_for_kind="header"),
            console=console,
            refresh_per_second=2,
            transient=True,
        )
        self._live = Live(
            Group(self.header, self._outer_progress, self._workers_progress),
            console=console,
            refresh_per_second=2,
            vertical_overflow="ellipsis",
            transient=True,
        )
        self._exit_stack: ExitStack | None = None
        self._outer_task_id: TaskID | None = None
        self._worker_rows: list[tuple[TaskID, TaskID]] = []
        self._row_index = 0
        self._worker_task_ids: dict[int, TaskID] = {}
        self._worker_detail_task_ids: dict[int, TaskID] = {}
        self._total_files = 0
        self._started = False

    def __enter__(self) -> "RichBenchmarkProgress":
        if not self._started:
            # Only the outer Live is entered. The Progress objects are NOT
            # entered: entering them would nest their internal Lives on
            # console._live_stack, and the outermost Live renders every Live in
            # the stack — so every frame would render twice. Non-started
            # Progress objects still render their tasks through the Group on
            # each Live refresh (the documented multi-Progress pattern).
            self._exit_stack = ExitStack()
            self._exit_stack.enter_context(self._live)
            # Pre-create the per-worker header/detail row pairs so the display
            # has a stable height before any events arrive.
            for i in range(self.n_workers):
                header_id = self._workers_progress.add_task(
                    f"worker {i}: idle",
                    total=1,
                    completed=0,
                    visible=True,
                    file="",
                    kind="header",
                )
                detail_id = self._workers_progress.add_task(
                    "↳",
                    total=1,
                    completed=0,
                    visible=True,
                    file="",
                    kind="detail",
                )
                self._worker_rows.append((header_id, detail_id))
            self._started = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._started and self._exit_stack is not None:
            self._exit_stack.__exit__(exc_type, exc_value, traceback)
            self._exit_stack = None
            self._started = False

    def on_condition_started(
        self,
        condition_name: str,
        overrides: dict,
        total_files: int,
        transcriber_names: list[str],
    ) -> None:
        """Begin a benchmark condition, growing the overall sweep bar."""
        if not self._started:
            self.__enter__()
        self._total_files = total_files
        if self._outer_task_id is None:
            self._outer_task_id = self._outer_progress.add_task("benchmark", total=0)
        assert self._outer_task_id is not None
        current_total = self._outer_progress.tasks[self._outer_task_id].total
        self._outer_progress.update(
            self._outer_task_id,
            total=current_total + total_files * len(transcriber_names),
        )
        self._update_header()

    def _describe_worker(self, event: WorkerEvent) -> str:
        """Build a row description: pid, condition, transcriber, stage, device.

        Values are escaped before wrapping in style tags because
        condition/transcriber come from user-authored benchmark YAML and may
        contain ``[``/``]`` characters that would otherwise raise
        ``MarkupError`` (or mis-render) when ``TextColumn`` parses the
        description as markup; pid/stage/device are escaped too for uniformity.
        """
        description = (
            f"pid [{_STYLE_PID}]{escape(str(event.worker_id))}[/{_STYLE_PID}]"
            f" · [{_STYLE_CONDITION}]{escape(event.condition)}[/{_STYLE_CONDITION}]"
        )
        if event.transcriber:
            description += (
                f" × [{_STYLE_TRANSCRIBER}]{escape(event.transcriber)}"
                f"[/{_STYLE_TRANSCRIBER}]"
            )
        if event.stage:
            description += (
                f" · [{_STYLE_STAGE}]{escape(event.stage)}[/{_STYLE_STAGE}]"
            )
        if event.transcriber in self.devices:
            description += (
                f" · [{_STYLE_DEVICE}]{escape(self.devices[event.transcriber])}"
                f"[/{_STYLE_DEVICE}]"
            )
        return description

    def on_worker_event(self, event: WorkerEvent) -> None:
        """Update the display for one event streamed from a worker."""
        if not self._started:
            self.__enter__()
        header_id, detail_id = self._assign_row(event.worker_id)
        if event.status == "start":
            self._workers_progress.update(
                header_id,
                description=self._describe_worker(event),
            )
            self._workers_progress.update(
                detail_id,
                total=self._total_files,
                file=escape(_format_file_field(event.midi_path)),
            )
        elif event.status == "stage":
            # Update description, and file/total where the stage actually
            # knows them. "render" fires exactly once per condition, before
            # any per-file events, so it doubles as this row's
            # condition-boundary marker and is the only place completed is
            # reset - "separate" (mid-condition) must never touch it.
            self._workers_progress.update(
                header_id,
                description=self._describe_worker(event),
            )
            fields: dict[str, Any] = {}
            if event.stage == "render":
                fields["file"] = ""  # clear any stale filename
                fields["total"] = self._total_files  # row shows M/N during render too
                fields["completed"] = 0  # new condition for this worker/row
            elif event.stage == "separate":
                fields["file"] = escape(
                    _format_file_field(event.midi_path)
                )  # real, known filename
            if fields:
                self._workers_progress.update(detail_id, **fields)
        elif event.status == "done":
            self._workers_progress.advance(detail_id)
            if self._outer_task_id is not None:
                self._outer_progress.advance(self._outer_task_id)
            if not event.ok:
                self._failed += 1
            self._update_header()
        else:
            logger.warning("Ignoring unknown worker event status %r", event.status)

    def on_condition_done(self, condition_name: str) -> None:
        """No-op: per-worker rows complete via :meth:`on_worker_event` advances."""
        pass

    def _assign_row(self, worker_id: int) -> tuple[TaskID, TaskID]:
        """Return the (header, detail) row TaskIDs for a worker pid, assigning
        them lazily on first use.

        The first event from a pid takes the next unused pre-created row pair;
        once a pid owns a pair it keeps it for the whole run.
        """
        header_id = self._worker_task_ids.get(worker_id)
        if header_id is not None:
            return (header_id, self._worker_detail_task_ids[worker_id])
        if self._row_index < len(self._worker_rows):
            header_id, detail_id = self._worker_rows[self._row_index]
            self._row_index += 1
        else:
            # All pre-created rows are already assigned to other pids; fall back
            # to the first row pair so we never crash (shouldn't happen with a
            # correctly-sized pool).
            header_id, detail_id = self._worker_rows[0]
        self._worker_task_ids[worker_id] = header_id
        self._worker_detail_task_ids[worker_id] = detail_id
        self._pids.add(worker_id)
        self._update_header()
        return (header_id, detail_id)

    def _update_header(self) -> None:
        """Rebuild the header in place so the Live group sees the change."""
        done = 0
        total = 0
        if self._outer_task_id is not None:
            outer_task = self._outer_progress.tasks[self._outer_task_id]
            done = int(outer_task.completed)
            total = int(outer_task.total) if outer_task.total is not None else 0
        header_text = f"benchmark · {done}/{total} files · {self.n_workers} workers"
        if self.devices:
            header_text += " · devices: " + ", ".join(
                sorted(set(self.devices.values()), key=str.casefold)
            )
        if self._pids:
            header_text += " · pids [" + ", ".join(
                str(pid) for pid in sorted(self._pids)
            ) + "]"
        rendered = Text(header_text, style="bold")
        if self._failed:
            rendered.append(Text.from_markup(f" [red]{self._failed} failed[/]"))
        self.header.plain = rendered.plain
        self.header.spans = rendered.spans
        self.header.style = rendered.style
