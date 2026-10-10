"""Shared pytest fixtures, plus a guard that records reads of the shipped config tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

#: Modules holding a process-global "numeric settings already applied" cache:
#: basic_pitch has its own, torch_support holds the one shared by every torch
#: backend. Resetting by name keeps the reset off the import path.
_NUMERIC_STATE_MODULES = (
    "sonitra.transcribe.basic_pitch",
    "sonitra.transcribe.torch_support",
)

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: The shipped configuration tree. Tests that are not part of the preset suite
#: are meant to build their config from fixtures, so a read under this root is
#: reported against the test that performed it.
_GUARDED_ROOT: Path = _REPO_ROOT / "config"

#: Audit events that mean "the filesystem was read at this path". Other events
#: (imports, sockets, stat calls) cannot name a file being opened and are not
#: worth the per-event cost of the hook.
_GUARDED_EVENTS: frozenset[str] = frozenset({"open", "os.scandir", "os.listdir"})

#: Report mode: offenders are listed in the terminal summary and nothing fails,
#: so a suite can be baselined first. Flipping this to ``True`` turns every
#: listed path into a failure of the test that read it.
_ENFORCE: bool = False

#: Audit-hook state. An audit hook is process-wide and cannot see pytest, so
#: the run loop publishes the test it is executing through these globals.
_ACTIVE: bool = False
_CURRENT_ID: str = ""

#: Test node id -> offending paths as repo-relative strings, de-duplicated in
#: first-seen order.
_OFFENDERS: dict[str, list[str]] = {}
_TRUNCATED: set[str] = set()

#: One runaway test must not be able to flood the summary.
_MAX_OFFENDERS_PER_TEST: int = 25


def _is_guarded(event: str, args: tuple[object, ...], root: Path) -> Path | None:
    """Return the resolved path an audit event reads under ``root``, else None."""
    if event not in _GUARDED_EVENTS:
        return None
    if not args or args[0] is None or isinstance(args[0], int):
        return None
    candidate = args[0]
    if not isinstance(candidate, (str, os.PathLike)):
        return None
    try:
        # Relative paths mean what they mean to the caller, so they are
        # resolved against the working directory the read would have used.
        resolved = Path(os.fsdecode(candidate)).resolve()
        base = Path(os.fsdecode(root)).resolve()
    except (OSError, TypeError, ValueError, RuntimeError):
        return None
    # Component comparison, not a string prefix: "config/" is inside the tree,
    # "configs/" and a sibling "audio/config/" are not.
    if resolved == base or base in resolved.parents:
        return resolved
    return None


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _audit_hook(event: str, args: tuple[object, ...]) -> None:
    # This runs for every audited event in the process, including from inside
    # unrelated libraries and during interpreter shutdown, so it must stay
    # cheap and must never raise into its caller.
    try:
        if not _ACTIVE or event not in _GUARDED_EVENTS:
            return
        hit = _is_guarded(event, args, _GUARDED_ROOT)
        if hit is None or not _CURRENT_ID:
            return
        shown = _repo_relative(hit)
        paths = _OFFENDERS.setdefault(_CURRENT_ID, [])
        if shown in paths:
            return
        if len(paths) >= _MAX_OFFENDERS_PER_TEST:
            _TRUNCATED.add(_CURRENT_ID)
            return
        paths.append(shown)
    except Exception:
        return


sys.addaudithook(_audit_hook)


@pytest.fixture(autouse=True)
def _isolate_numeric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Numeric settings travel from the config to the backend builders through
    # process environment variables, so a test that sets one must never leak it
    # into the next test.
    monkeypatch.delenv("SONITRA_NUMERIC_MODE", raising=False)
    monkeypatch.delenv("SONITRA_GPU_MEMORY_GROWTH", raising=False)
    # The frameworks refuse repeated setup within one process, so each backend
    # caches what it already applied. A test that applies settings must not let
    # the next test inherit that cache; only already-imported modules are reset,
    # so the reset never pulls a backend onto the import path.
    for name in _NUMERIC_STATE_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        monkeypatch.setattr(module, "_NUMERIC_STATE", None, raising=False)
    # A pool worker outlives any single test, so a cached transcriber must not
    # survive into the next one; only an already-imported runner is touched.
    if "sonitra.benchmark.runner" in sys.modules:
        runner_module = sys.modules["sonitra.benchmark.runner"]
        monkeypatch.setattr(runner_module, "_WORKER_TRANSCRIBERS", {})


@pytest.fixture
def midi_fixture() -> callable:
    fixtures_dir = Path(__file__).parent / "fixtures"

    def _fixture(name: str) -> Path:
        return fixtures_dir / name

    return _fixture


@pytest.fixture
def config_fixture() -> callable:
    fixtures_dir = Path(__file__).parent / "fixtures"

    def _fixture(name: str) -> Path:
        return fixtures_dir / name

    return _fixture


@pytest.fixture
def corpus_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def session_engine():
    from sonitra.engine import RendererEngine

    return RendererEngine(sample_rate=44100, block_size=512)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def vst_path() -> Path:
    path = os.getenv("VST_PATH") or os.getenv("VST3_PATH")
    if not path:
        pytest.skip("VST plugin path not configured")
    return Path(path)


@pytest.fixture
def vital_vst_path() -> Path:
    path = Path("/workspace/plugin/vital/lib/vst3/Vital.vst3")
    if not path.exists():
        pytest.skip("Vital VST3 not found at /workspace/plugin/vital/lib/vst3/Vital.vst3")
    return path


@pytest.fixture
def dummy_audio() -> np.ndarray:
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    return np.stack([signal, signal])


@pytest.fixture
def dummy_silent_audio() -> np.ndarray:
    return np.zeros((2, 44100), dtype=np.float32)


@pytest.fixture
def audio_corpus_dir(tmp_path: Path) -> Path:
    """Tmp corpus tree with ``midi/`` reference files and ``recordings/`` audio.

    Recordings are named ``piece_<n>_<performer>.wav`` so they pair to their
    reference ``piece_<n>.mid`` by the token-prefix rule.
    """
    import shutil

    from sonitra.storage import write_wav

    fixtures_dir = Path(__file__).parent / "fixtures"
    midi_dir = tmp_path / "midi"
    recordings_dir = tmp_path / "recordings"
    midi_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(fixtures_dir / "test_c4.mid", midi_dir / "piece_0.mid")
    shutil.copy(fixtures_dir / "test_polyphonic.mid", midi_dir / "piece_1.mid")

    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    for index, freq in ((0, 440.0), (1, 523.25)):
        signal = 0.25 * np.sin(2 * np.pi * freq * t)
        tone = np.stack([signal, signal])
        write_wav(
            tone,
            recordings_dir / f"piece_{index}_performer1.wav",
            sample_rate=sample_rate,
            normalize=False,
        )

    return tmp_path


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    from sonitra.storage import write_wav

    path = tmp_path / "silent.wav"
    write_wav(np.zeros((2, 44100), dtype=np.float32), path, sample_rate=44100)
    return path


@pytest.fixture
def off_rate_wav(tmp_path: Path) -> Path:
    """A WAV written at 48000Hz while the test config's ``pipeline.sample_rate``
    stays at the 44100Hz default — the regression fixture for audio-mode
    output following the source file's own rate, not the config rate."""
    from sonitra.storage import write_wav

    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    tone = np.stack([signal, signal])
    path = tmp_path / "off_rate.wav"
    write_wav(tone, path, sample_rate=sample_rate, normalize=False)
    return path


def pytest_runtest_setup(item: pytest.Item) -> None:
    global _ACTIVE, _CURRENT_ID
    if "skip_if_no_vst" in item.keywords:
        path = os.getenv("VST_PATH") or os.getenv("VST3_PATH")
        if not path:
            pytest.skip("VST plugin path not configured")
    # Reading the shipped configs is the point of the preset suite; everywhere
    # else it couples a test to files it is meant to be independent of.
    _ACTIVE = "presets" not in item.keywords
    _CURRENT_ID = item.nodeid


def pytest_runtest_call(item: pytest.Item) -> None:
    global _ACTIVE, _CURRENT_ID
    # Re-asserted after setup so a shared fixture's read is charged to the test
    # that asked for it rather than to whoever ran before.
    _ACTIVE = "presets" not in item.keywords
    _CURRENT_ID = item.nodeid


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    global _ACTIVE
    # Cleared before fixtures are torn down so a read during teardown is not
    # charged to the test that happens to run next.
    _ACTIVE = False
    offenders = _OFFENDERS.get(item.nodeid)
    if _ENFORCE and offenders:
        pytest.fail("read the shipped config tree:\n  " + "\n  ".join(offenders))


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    if not _OFFENDERS:
        return
    total = sum(len(paths) for paths in _OFFENDERS.values())
    terminalreporter.write_sep(
        "=", f"config/ reads outside the preset tests: {len(_OFFENDERS)} tests, {total} paths"
    )
    for test_id in sorted(_OFFENDERS):
        terminalreporter.write_line(f"  {test_id}")
        for path in _OFFENDERS[test_id]:
            terminalreporter.write_line(f"    {path}")
        if test_id in _TRUNCATED:
            terminalreporter.write_line(f"    ... more than {_MAX_OFFENDERS_PER_TEST} paths")
    if not _ENFORCE:
        terminalreporter.write_line("  (report only; nothing was failed)")
