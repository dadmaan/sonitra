#!/usr/bin/env python3
"""Download AMT benchmark datasets into the Sonitra corpus directory.

This script is intentionally self-contained: it runs on the Python stdlib
alone, so it can be used before the project environment is set up. When
``rich`` happens to be installed *and* stdout is an interactive terminal, an
interactive dataset picker and a live download display are offered instead;
without rich or a TTY the script falls back to exactly its plain stdlib
behavior.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py maestro-v3
    python scripts/download_datasets.py bsed
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --all --jobs 4
    python scripts/download_datasets.py maestro-v3 --output-dir /data/corpus
    python scripts/download_datasets.py            # interactive picker (TTY only)

Interactive mode: run with no dataset name and no --all on a terminal with
`rich` installed, pick any number of datasets from the table, and download up
to --jobs of them concurrently.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

try:
    from rich.console import Console as _RichConsole, Group as _RichGroup
    from rich.live import Live as _RichLive
    from rich.progress import (
        BarColumn as _BarColumn,
        DownloadColumn as _DownloadColumn,
        MofNCompleteColumn as _MofNCompleteColumn,
        Progress as _RichProgress,
        TextColumn as _TextColumn,
        TimeElapsedColumn as _TimeElapsedColumn,
        TimeRemainingColumn as _TimeRemainingColumn,
        TransferSpeedColumn as _TransferSpeedColumn,
    )
    from rich.prompt import Prompt as _RichPrompt
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText
except ImportError:  # pragma: no cover - exercised in rich-less environments
    _HAS_RICH = False
else:
    _HAS_RICH = True

REPO: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# To add a new dataset: copy one entry below and fill in the fields.
# `extract_map` is a list of (zip_source_prefix, target_subdir) pairs: any zip
# member whose path starts with a given prefix is extracted (with that prefix
# stripped) under corpus_subdir/target_subdir/. Members matching no prefix are
# skipped. Most datasets need a single ("<top-level-dir>/", "midi") pair;
# multi-modal datasets (e.g. MIDI + audio) can route different zip subfolders
# to different target subdirs.
DATASETS: Dict[str, Dict] = {
    "maestro-v3": {
        "name": "MAESTRO V3.0.0 (MIDI only)",
        "description": (
            "1,276 piano MIDI files paired with professional recordings. "
            "Standard AMT benchmark (Hawthorne et al., ICLR 2019)."
        ),
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "corpus_subdir": "maestro-v3",
        "extract_map": [("maestro-v3.0.0/", "midi")],
        "size_mb": 57,  # ~57 MB: the MIDI-only zip, unpacked
    },
    "bsed": {
        "name": "Beethoven Symphony Excerpt Dataset (BSED) v1.0",
        "description": (
            "20 Beethoven symphony excerpts: MIDI scores paired with 4 real concert "
            "recordings + 1 synthetic rendition each (pitch-corrected to A440). "
            "Note-level score-audio alignment annotations and MusicXML/Sibelius scores "
            "are also on Zenodo but not fetched by this script — see ROADMAP.md. "
            "CC BY-NC-SA 4.0 (noncommercial). Berendes et al., TISMIR 2026."
        ),
        "url": "https://zenodo.org/records/20344500/files/BSED.zip",
        "corpus_subdir": "bsed",
        "extract_map": [
            ("BSED_1.0/01_ScoreData/MIDI/", "midi"),
            ("BSED_1.0/02_Audio/wav_44100_440Hz/", "recordings"),
        ],
        "size_mb": 380,  # ~380 MB: MIDI + 4 real concert recordings + 1 synthetic rendition per excerpt
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_dirs(output_dir: Path, spec: Dict) -> List[Path]:
    """Return one path per distinct target subdir named in spec's extract_map."""
    dataset_dir = output_dir / spec["corpus_subdir"]
    subdirs = sorted({target_subdir for _, target_subdir in spec["extract_map"]})
    return [dataset_dir / subdir for subdir in subdirs]


def _is_already_present(output_dir: Path, spec: Dict) -> bool:
    """Return True when every target subdir exists and contains at least one file."""
    dirs = _target_dirs(output_dir, spec)
    return all(d.is_dir() and any(d.iterdir()) for d in dirs)


def _print_list(output_dir: Path) -> None:
    col_name = max(len(k) for k in DATASETS) + 2
    print(f"{'Dataset':<{col_name}}  {'Target path':<40}  Description")
    print("-" * 120)
    for key, spec in DATASETS.items():
        target = output_dir / spec["corpus_subdir"]
        print(f"{key:<{col_name}}  {str(target):<40}  {spec['description']}")


def _make_reporthook(name: str):
    """Return a urlretrieve reporthook that prints download progress in-place."""

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded: float = block_num * block_size
        if total_size > 0:
            total_mb: float = total_size / 1_048_576
            done_mb: float = min(downloaded, total_size) / 1_048_576
            print(
                f"\rDownloading {name}: {done_mb:.1f} MB / {total_mb:.1f} MB",
                end="",
                flush=True,
            )
        else:
            done_mb = downloaded / 1_048_576
            print(
                f"\rDownloading {name}: {done_mb:.1f} MB",
                end="",
                flush=True,
            )

    return reporthook


def _extract_zip(tmp_path: str, spec: Dict, output_dir: Path) -> int:
    """Extract a downloaded zip according to spec's extract_map. Returns file count."""
    dataset_dir: Path = output_dir / spec["corpus_subdir"]
    extract_map: List[Tuple[str, str]] = spec["extract_map"]

    files_extracted: int = 0
    with zipfile.ZipFile(tmp_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            for src_prefix, target_subdir in extract_map:
                if not member.filename.startswith(src_prefix):
                    continue
                relative: str = member.filename[len(src_prefix) :]
                if not relative:
                    break
                dest: Path = dataset_dir / target_subdir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                files_extracted += 1
                break
    return files_extracted


def _download_and_extract(key: str, spec: Dict, output_dir: Path) -> int:
    """Download and extract one dataset (plain stdlib path). Returns file count."""
    name: str = spec["name"]

    tmp_path: str = ""
    tmp_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path = tmp_fd.name
    tmp_fd.close()

    try:
        urlretrieve(spec["url"], tmp_path, reporthook=_make_reporthook(name))
        print()  # newline after the progress line
        return _extract_zip(tmp_path, spec, output_dir)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _parse_selection(response: str, keys: List[str]) -> List[str]:
    """Parse an interactive picker response into an ordered list of dataset keys.

    Accepts comma-separated 1-based indices (``"1,2"``), ``"all"`` for every
    key, or ``"q"``/empty input to quit (returns ``[]``). Duplicate selections
    are collapsed. Raises ``ValueError`` for any invalid or out-of-range token.
    """
    text: str = response.strip()
    lowered: str = text.lower()
    if lowered == "all":
        return list(keys)
    if lowered == "q" or text == "":
        return []
    selected: List[str] = []
    seen: set = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            raise ValueError(
                f"invalid selection '{response}': empty item in list"
            )
        try:
            index = int(token)
        except ValueError:
            raise ValueError(
                f"invalid selection '{response}': '{token}' is not a number"
            )
        if index < 1 or index > len(keys):
            raise ValueError(
                f"invalid selection '{response}': {index} is out of range "
                f"(expected 1-{len(keys)})"
            )
        key = keys[index - 1]
        if key not in seen:
            seen.add(key)
            selected.append(key)
    if not selected:
        raise ValueError(f"invalid selection '{response}'")
    return selected


def _print_table(console: "_RichConsole", output_dir: Path) -> None:
    """Render the dataset registry as a rich table (--list and the picker share it)."""
    table = _RichTable(title="Available datasets", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("key")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("target")
    table.add_column("status")
    for index, (key, spec) in enumerate(DATASETS.items(), start=1):
        target = output_dir / spec["corpus_subdir"]
        badge = "present" if _is_already_present(output_dir, spec) else "missing"
        table.add_row(
            str(index),
            key,
            spec["name"],
            f"{spec['size_mb']} MB",
            str(target),
            f"[green]present[/green]" if badge == "present" else "[yellow]missing[/yellow]",
        )
    console.print(table)


def _interactive_select(output_dir: Path) -> Optional[List[str]]:
    """Show the picker table and prompt for a comma-separated multi-select.

    Returns the selected dataset keys, or ``None`` when the user quits
    (``"q"``/empty input/EOF).
    """
    console = _RichConsole()
    keys: List[str] = list(DATASETS.keys())
    while True:
        _print_table(console, output_dir)
        try:
            response = _RichPrompt.ask(
                "Select datasets (comma-separated numbers, 'all', or 'q' to quit)"
            )
        except EOFError:
            return None
        try:
            selected = _parse_selection(response, keys)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        if not selected:
            return None
        return selected


class _DownloadDisplay:
    """Live rich download display (rich + TTY only).

    A bold aggregate header (``download · N/M datasets · X MB · Y MB/s``, with
    a ``[red]N failed[/]`` suffix when applicable) above one progress row per
    worker slot. Thread-safe: all mutable state is guarded by an RLock, and
    rich's Progress is itself RLock-protected.
    """

    def __init__(
        self,
        console: "_RichConsole",
        *,
        slots: int,
        total_datasets: int,
        total_mb: float,
    ) -> None:
        self._lock = threading.RLock()
        self._total_datasets = total_datasets
        self._total_mb = total_mb
        self._completed = 0
        self._failed = 0
        self._acc_bytes = 0.0  # bytes from datasets whose slots already finished
        self._started_at = time.monotonic()
        self._progress = _RichProgress(
            _TextColumn("{task.description}"),
            _DownloadColumn(),
            _TransferSpeedColumn(),
            _BarColumn(),
            _MofNCompleteColumn(),
            _TimeElapsedColumn(),
            _TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
            transient=True,
        )
        # One row per worker slot, pre-created so the display keeps a stable
        # height. Only the outer Live is entered — entering the Progress would
        # nest its internal Live and double-render every frame.
        self.tasks: List[int] = [
            self._progress.add_task(
                f"slot {i}: idle", total=1, completed=0, visible=True
            )
            for i in range(slots)
        ]
        self._live = _RichLive(
            console=console,
            get_renderable=self._get_renderable,
            refresh_per_second=2,
            vertical_overflow="ellipsis",
        )

    def __enter__(self) -> "_DownloadDisplay":
        self._live.start(refresh=True)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        self._live.stop()

    def _get_renderable(self) -> "_RichGroup":
        """Rebuild the header + progress group on every Live refresh."""
        with self._lock:
            completed = self._completed
            failed = self._failed
            live_bytes = self._acc_bytes + sum(
                min(t.completed, t.total or 0) for t in self._progress.tasks
            )
            elapsed = max(time.monotonic() - self._started_at, 0.0001)
        header = _RichText(
            f"download · {completed}/{self._total_datasets} datasets · "
            f"{self._total_mb:.1f} MB · {live_bytes / elapsed / 1_048_576:.1f} MB/s",
            style="bold",
        )
        if failed:
            header.append(_RichText.from_markup(f" [red]{failed} failed[/]"))
        return _RichGroup(header, self._progress)

    def start_task(self, task_id: int, name: str, size_hint: int) -> None:
        """Assign a slot row to a new dataset download."""
        with self._lock:
            self._progress.update(
                task_id,
                description=name,
                total=size_hint if size_hint > 0 else None,
                completed=0,
            )

    def on_progress(self, task_id: int, downloaded: int, total_size: int) -> None:
        """Feed a urlretrieve reporthook call into the slot's task."""
        with self._lock:
            if total_size > 0:
                self._progress.update(
                    task_id,
                    completed=min(downloaded, total_size),
                    total=total_size,
                )
            else:
                self._progress.update(task_id, completed=downloaded)

    def on_extracting(self, task_id: int, name: str) -> None:
        """Switch a fully-downloaded slot row to the dim 'extracting…' phase."""
        with self._lock:
            self._progress.update(
                task_id, description=f"[dim]extracting {name}…[/dim]"
            )

    def finish_task(self, task_id: int, name: str, status: str) -> None:
        """Mark a slot row done/skipped/errored and count it in the header."""
        with self._lock:
            task = self._progress.tasks[task_id]
            self._acc_bytes += min(task.completed, task.total or 0)
            self._completed += 1
            if status == "error":
                self._failed += 1
            self._progress.update(task_id, description=f"{name} · {status}")

    def any_failure(self) -> bool:
        with self._lock:
            return self._failed > 0


def _download_one(
    key: str,
    spec: Dict,
    output_dir: Path,
    *,
    display: Optional["_DownloadDisplay"] = None,
    task_id: Optional[int] = None,
) -> Tuple[str, int, Optional[str]]:
    """Download and extract one dataset.

    Returns a ``(status, files_extracted, error)`` triple with ``status`` one
    of ``"skip"`` (already present), ``"done"``, or ``"error"``. When
    ``display`` is given the dataset's progress is rendered into its slot row.
    """
    name: str = spec["name"]
    if _is_already_present(output_dir, spec):
        if display is not None:
            display.finish_task(task_id, name, "skip")
        return "skip", 0, None
    try:
        if display is not None:
            display.start_task(
                task_id, name, int(spec.get("size_mb", 0)) * 1_048_576
            )
            n_files: int = _download_and_extract_rich(spec, output_dir, display, task_id)
        else:
            n_files = _download_and_extract(key, spec, output_dir)
    except Exception as exc:  # noqa: BLE001 - per-dataset failure isolation
        if display is not None:
            display.finish_task(task_id, name, "error")
        return "error", 0, str(exc)
    if display is not None:
        display.finish_task(task_id, name, "done")
    return "done", n_files, None


def _download_and_extract_rich(
    spec: Dict, output_dir: Path, display: "_DownloadDisplay", task_id: int
) -> int:
    """Rich path: download into the display's slot row, then extract."""
    name: str = spec["name"]
    tmp_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path: str = tmp_fd.name
    tmp_fd.close()

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        display.on_progress(task_id, block_num * block_size, total_size)

    try:
        urlretrieve(spec["url"], tmp_path, reporthook=reporthook)
        display.on_extracting(task_id, name)
        return _extract_zip(tmp_path, spec, output_dir)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _slot_worker(
    slot: int,
    task_id: int,
    work_queue: "queue.Queue[str]",
    display: "_DownloadDisplay",
    output_dir: Path,
) -> bool:
    """Drain the shared queue, one dataset per slot row. Returns True on error."""
    any_failure: bool = False
    while True:
        try:
            key = work_queue.get_nowait()
        except queue.Empty:
            return any_failure
        spec = DATASETS[key]
        status, _n_files, _error = _download_one(
            key, spec, output_dir, display=display, task_id=task_id
        )
        if status == "error":
            any_failure = True


def _run_rich(
    selected: List[str], output_dir: Path, jobs: int, display: "_DownloadDisplay"
) -> bool:
    """Download selected datasets through the rich display. Returns True on error."""
    work_queue: "queue.Queue[str]" = queue.Queue()
    for key in selected:
        work_queue.put(key)
    slots = min(jobs, len(selected))
    executor = ThreadPoolExecutor(max_workers=slots)
    futures = [
        executor.submit(
            _slot_worker, slot, display.tasks[slot], work_queue, display, output_dir
        )
        for slot in range(slots)
    ]
    try:
        with display:
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return display.any_failure() or any(f.result() for f in futures)


def _run_plain(selected: List[str], output_dir: Path, jobs: int) -> bool:
    """Download selected datasets with the plain stdlib output. Returns True on error."""
    any_failure: bool = False

    def worker(key: str) -> bool:
        spec = DATASETS[key]
        dataset_dir: Path = output_dir / spec["corpus_subdir"]
        status, n_files, error = _download_one(key, spec, output_dir)
        if status == "skip":
            print(f"[skip] {spec['name']} — already present at {dataset_dir}")
        elif status == "done":
            print(f"[done] {spec['name']}  {n_files} files extracted -> {dataset_dir}")
        else:
            print(f"[error] {spec['name']}: {error}", file=sys.stderr)
        return status == "error"

    if jobs <= 1:
        for key in selected:
            if worker(key):
                any_failure = True
        return any_failure

    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = [executor.submit(worker, key) for key in selected]
    try:
        results = [future.result() for future in futures]
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return any(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _can_interact() -> bool:
    """True when the interactive picker can run: rich installed + a real TTY."""
    return _HAS_RICH and sys.stdin.isatty() and sys.stdout.isatty()


def _use_rich_output() -> bool:
    """True when rich output (table / live display) may be used."""
    return _HAS_RICH and sys.stdout.isatty()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AMT benchmark datasets into the Sonitra corpus directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Examples:",
                "  python scripts/download_datasets.py --list",
                "  python scripts/download_datasets.py maestro-v3",
                "  python scripts/download_datasets.py bsed",
                "  python scripts/download_datasets.py --all",
                "  python scripts/download_datasets.py --all --jobs 4",
                "  python scripts/download_datasets.py  (interactive picker on a TTY)",
            ]
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "dataset",
        nargs="?",
        metavar="DATASET",
        help="Name of dataset to download (see --list for available names).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available datasets and exit.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Download up to N datasets concurrently (default: 1 = serial).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Root corpus directory (default: <repo>/corpus). "
            "Dataset files are placed under OUTPUT_DIR/<corpus_subdir>/midi/."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir: Path = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else REPO / "corpus"
    )

    if args.list:
        if _use_rich_output():
            _print_table(_RichConsole(), output_dir)
        else:
            _print_list(output_dir)
        return 0

    if args.dataset is None and not args.all:
        if _can_interact():
            selected = _interactive_select(output_dir)
            if selected is None:
                return 0
        else:
            print(
                "error: specify a dataset name or pass --all. "
                "Use --list to see available datasets.",
                file=sys.stderr,
            )
            return 1
    elif args.all:
        selected = list(DATASETS.keys())
    else:
        if args.dataset not in DATASETS:
            available: str = ", ".join(DATASETS.keys())
            print(
                f"error: unknown dataset '{args.dataset}'. Available: {available}",
                file=sys.stderr,
            )
            return 1
        selected = [args.dataset]

    try:
        if _use_rich_output() and selected:
            display = _DownloadDisplay(
                _RichConsole(),
                slots=min(args.jobs, len(selected)),
                total_datasets=len(selected),
                total_mb=sum(DATASETS[k]["size_mb"] for k in selected),
            )
            any_failure = _run_rich(selected, output_dir, args.jobs, display)
        else:
            any_failure = _run_plain(selected, output_dir, args.jobs)
    except KeyboardInterrupt:
        if _use_rich_output():
            _RichConsole().print("[yellow]Interrupted — partial results kept[/yellow]")
        else:
            print("Interrupted — partial results kept", file=sys.stderr)
        return 130

    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
