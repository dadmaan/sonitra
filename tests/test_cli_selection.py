from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from sonitra.cli import _apply_dataset, _apply_selection_overrides, _apply_subset, app
from sonitra.config import PipelineConfig, SelectionSample
from sonitra.selection import SelectionError, sample_units

FIXTURES = Path(__file__).parent / "fixtures"
SOUNDFONT = "/usr/share/sounds/sf2/default-GM.sf2"
DATASET = "mini"


@pytest.fixture(autouse=True)
def _reset_console_singleton():
    """Reset the process-wide Console singleton around every test.

    ``sonitra.terminal.get_console()`` caches its Console; a CliRunner
    invocation closes the stream the Console was pinned to, so a later
    invocation in the same process would write to a dead stream. Reset
    before and after each test, matching tests/test_cli_audio.py.
    """
    import sonitra.terminal as terminal_module

    terminal_module._console = None
    yield
    terminal_module._console = None


def _copy_midi(stem: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.mid"
    shutil.copy(FIXTURES / "test_c4.mid", path)
    return path


def _write_metadata(
    root: Path,
    rows: list[dict[str, str]],
    *,
    dataset: str = DATASET,
    name: str = "meta.csv",
) -> Path:
    path = root / dataset / "metadata" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _selection_block(
    where: dict[str, list[str]] | None = None,
    *,
    sample: dict[str, int] | None = None,
    metadata_csv: str = "meta.csv",
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if where:
        block["metadata_csv"] = metadata_csv
        block["where"] = where
    if sample is not None:
        block["sample"] = sample
    return block


def _config_payload(
    corpus_root: Path,
    *,
    dataset: str | None = None,
    selection: dict[str, Any] | None = None,
    input_type: str = "midi",
    file_naming: str = "{stem}",
    frame_metrics: bool = True,
    benchmark_resume: bool = False,
    midi_dir: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "render_pipeline": {
            "synth_backend": "fluidsynth",
            "effects_chain": "none",
            "input_type": input_type,
            "bpm": 120,
            "sample_rate": 22050,
            "bit_depth": 16,
            "channels": 1,
            "duration_padding_sec": 0.5,
            "overwrite": True,
            "resume": False,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": {
            "corpus_root": str(corpus_root),
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": file_naming,
        },
        "fluidsynth": {"soundfont_path": SOUNDFONT},
        "observability": {
            "write_manifest": False,
            "manifest_path": "./renders.jsonl",
            "write_failed_list": False,
            "emit_sse_events": False,
            "progress": False,
        },
        "transcription": {
            "transcribers": [
                {"type": "precomputed", "name": "oracle", "midi_dir": str(midi_dir)}
            ]
        },
        "evaluation": {"frame_metrics": {"enabled": frame_metrics}},
        "benchmark": {"resume": benchmark_resume},
    }
    if dataset is not None:
        payload["io"]["dataset"] = dataset
    if selection is not None:
        payload["io"].update(selection)
        if selection.get("where"):
            # A filter needs a dataset to find its metadata CSV.
            payload["io"].setdefault("dataset", DATASET)
    return payload


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# evaluate always loads its config, like every other command
# ---------------------------------------------------------------------------


def test_evaluate_reads_default_config_yaml_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref_dir = tmp_path / "ref"
    est_dir = tmp_path / "est"
    _copy_midi("piece_0", ref_dir)
    _copy_midi("piece_0", est_dir)

    workdir = tmp_path / "work"
    workdir.mkdir()
    _write_config(
        workdir / "config.yaml",
        _config_payload(tmp_path / "data-root", frame_metrics=False, midi_dir=est_dir),
    )
    monkeypatch.chdir(workdir)

    output = tmp_path / "results.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--reference", str(ref_dir),
            "--estimate", str(est_dir),
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert any(key.startswith("note.") for key in rows[0])
    assert not any(key.startswith("frame.") for key in rows[0])


def test_evaluate_dataset_paths_follow_io_corpus_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deliberately not named "corpus": HEAD hard-codes ./corpus/<dataset>/...
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    _copy_midi("piece_0", midi_dir)
    est_dir = root / DATASET / "transcription" / "config" / "oracle"
    _copy_midi("piece_0", est_dir)

    workdir = tmp_path / "work"
    workdir.mkdir()
    _write_config(
        workdir / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    monkeypatch.chdir(workdir)

    output = tmp_path / "results.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["evaluate", "--dataset", DATASET, "--output", str(output)],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1


def test_evaluate_missing_config_matches_render_behaviour(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    ref_dir = tmp_path / "ref"
    _copy_midi("piece_0", ref_dir)

    runner = CliRunner()
    evaluate_result = runner.invoke(
        app,
        [
            "evaluate",
            "--config", str(missing),
            "--reference", str(ref_dir),
            "--estimate", str(ref_dir),
        ],
    )
    render_result = runner.invoke(
        app,
        ["render", "--config", str(missing), "--corpus", str(ref_dir)],
    )
    assert evaluate_result.exit_code == render_result.exit_code == 1
    assert "Config not found" in evaluate_result.stderr
    assert evaluate_result.stderr == render_result.stderr


def test_evaluate_limit_is_written_into_config_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sonitra.selection as selection_module

    ref_dir = tmp_path / "ref"
    est_dir = tmp_path / "est"
    for index in range(3):
        _copy_midi(f"piece_{index}", ref_dir)
        _copy_midi(f"piece_{index}", est_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(tmp_path / "data-root", midi_dir=est_dir),
    )

    captured: dict[str, Any] = {}
    real_select_references = selection_module.select_references

    def _spy(cfg: PipelineConfig, references: Any, **kwargs: Any) -> Any:
        captured["sample"] = cfg.io.sample
        return real_select_references(cfg, references, **kwargs)

    monkeypatch.setattr(selection_module, "select_references", _spy)

    output = tmp_path / "results.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--reference", str(ref_dir),
            "--estimate", str(est_dir),
            "--output", str(output),
            "--limit", "1",
        ],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
    sample = captured["sample"]
    assert sample is not None
    assert sample.n == 1
    assert sample.seed == 0


# ---------------------------------------------------------------------------
# --limit/--seed are written into io.sample
# ---------------------------------------------------------------------------


def _override_config(
    tmp_path: Path, selection: dict[str, Any] | None = None
) -> PipelineConfig:
    return PipelineConfig.model_validate(
        _config_payload(tmp_path / "data-root", selection=selection, midi_dir=tmp_path)
    )


def test_limit_creates_selection_block_when_none(tmp_path: Path) -> None:
    cfg = _override_config(tmp_path)
    _apply_selection_overrides(cfg, 2, None)
    assert cfg.io.sample == SelectionSample(n=2, seed=0)
    assert cfg.io.where == {}
    assert cfg.io.dataset is None


def test_limit_adds_sample_to_block_without_sample(tmp_path: Path) -> None:
    cfg = _override_config(tmp_path, _selection_block({"split": ["test"]}))
    _apply_selection_overrides(cfg, 2, 5)
    assert cfg.io.sample == SelectionSample(n=2, seed=5)
    assert cfg.io.where == {"split": ["test"]}
    assert cfg.io.dataset == DATASET


def test_limit_replaces_existing_sample(tmp_path: Path) -> None:
    cfg = _override_config(
        tmp_path,
        _selection_block({"split": ["test"]}, sample={"n": 5, "seed": 9}),
    )
    _apply_selection_overrides(cfg, 2, None)
    assert cfg.io.sample == SelectionSample(n=2, seed=0)
    assert cfg.io.where == {"split": ["test"]}


def test_seed_only_updates_existing_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _override_config(tmp_path, {"sample": {"n": 3, "seed": 0}})
    _apply_selection_overrides(cfg, None, 7)
    assert cfg.io.sample == SelectionSample(n=3, seed=7)
    assert "warning" not in capsys.readouterr().err


def test_seed_only_without_sample_warns_and_leaves_config_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _override_config(tmp_path)
    _apply_selection_overrides(cfg, None, 7)
    assert cfg.io.sample is None
    assert (
        "warning: --seed has no effect without --limit or io.sample"
        in capsys.readouterr().err
    )

    with_where = _override_config(tmp_path, _selection_block({"split": ["test"]}))
    before = with_where.io.model_copy(deep=True)
    _apply_selection_overrides(with_where, None, 7)
    assert with_where.io == before
    assert with_where.io.sample is None
    assert (
        "warning: --seed has no effect without --limit or io.sample"
        in capsys.readouterr().err
    )


def _legacy_apply_subset(
    files: list[Path], limit: int | None, seed: int | None
) -> list[Path]:
    import random

    if limit is None or limit >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, limit))


def test_apply_subset_alias_unchanged_for_seeded_calls() -> None:
    files = sorted(Path(f"file_{index:02d}.mid") for index in range(10))
    for seed in (0, 1, 123):
        for n in (1, 2, 5):
            assert _apply_subset(files, n, seed) == _legacy_apply_subset(files, n, seed)
    assert _apply_subset(files, None, 0) == files


def test_limit_zero_is_E11_exit_1(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi"
    _copy_midi("piece_0", midi_dir)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(tmp_path / "data-root", midi_dir=midi_dir),
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "render",
            "--config", str(config_path),
            "--corpus", str(midi_dir),
            "--output", str(out_dir),
            "--limit", "0",
        ],
    )
    assert result.exit_code == 1, f"render failed:\n{result.output}"
    assert "error: invalid selection from --limit/--seed" in result.stderr
    assert not out_dir.exists()


def test_override_result_is_validated(tmp_path: Path) -> None:
    cfg = _override_config(tmp_path, {"sample": {"n": 1, "seed": 0}})
    with pytest.raises(SelectionError, match="invalid selection from --limit/--seed"):
        _apply_selection_overrides(cfg, None, "not-an-int")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render/transcribe/evaluate/benchmark resolve their file lists
# ---------------------------------------------------------------------------

_SPLIT_ROWS = [
    {"midi_filename": "train_a.mid", "split": "train"},
    {"midi_filename": "test_a.mid", "split": "test"},
    {"midi_filename": "test_b.mid", "split": "test"},
]


def _make_corpus(
    tmp_path: Path, rows: list[dict[str, str]]
) -> tuple[Path, Path]:
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    for row in rows:
        _copy_midi(Path(row["midi_filename"]).stem, midi_dir)
    _write_metadata(root, rows)
    return root, midi_dir


def _write_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return path


class _FakePipelineResult:
    def __init__(self, succeeded: int) -> None:
        self.succeeded = succeeded
        self.failed = 0
        self.skipped = 0
        self.elapsed_seconds = 0.0


def _patch_run_pipeline(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    import sonitra.pipeline as pipeline_module

    def _fake_run_pipeline(
        midi_paths: Any,
        out_dir: Any = None,
        config: Any = None,
        corpus_root: Any = None,
        on_file_done: Any = None,
    ) -> _FakePipelineResult:
        captured["files"] = sorted(Path(path) for path in midi_paths)
        captured["calls"] = captured.get("calls", 0) + 1
        return _FakePipelineResult(len(captured["files"]))

    monkeypatch.setattr(pipeline_module, "run_pipeline", _fake_run_pipeline)


def _patch_run_benchmark(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    from sonitra.benchmark import runner as runner_module

    def _fake_run_benchmark(
        midi_paths: Any,
        work_dir: Any,
        config: Any,
        corpus_root: Any = None,
        *,
        audio_paths: Any = None,
        progress: Any = None,
        selection: Any = None,
    ) -> Any:
        captured["midi_paths"] = sorted(Path(path) for path in midi_paths)
        captured["audio_paths"] = (
            sorted(Path(path) for path in audio_paths)
            if audio_paths is not None
            else None
        )
        captured["config"] = config
        captured["selection"] = selection
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner_module.BenchmarkResult(
            records=[],
            summary=[],
            degradation=[],
            results_path=work_dir / "results.jsonl",
            summary_path=work_dir / "summary.json",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "run_benchmark", _fake_run_benchmark)


# ── render ──────────────────────────────────────────────────────────


def test_render_selection_filters_to_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    captured: dict[str, Any] = {}
    _patch_run_pipeline(monkeypatch, captured)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "render",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"
    assert [path.stem for path in captured["files"]] == ["test_a", "test_b"]
    assert "selection: mini split=test -> 2/3 files" in result.stdout


def test_render_limit_is_deterministic_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    for index in range(8):
        _copy_midi(f"piece_{index}", midi_dir)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )

    expected = sample_units(
        sorted(midi_dir.glob("*.mid")), SelectionSample(n=3, seed=0)
    )

    runs: list[list[Path]] = []
    for index in range(2):
        captured: dict[str, Any] = {}
        _patch_run_pipeline(monkeypatch, captured)
        result = CliRunner().invoke(
            app,
            [
                "render",
                "--config", str(config_path),
                "--corpus", str(midi_dir),
                "--output", str(tmp_path / f"out_{index}"),
                "--limit", "3",
            ],
        )
        assert result.exit_code == 0, f"render failed:\n{result.output}"
        runs.append(captured["files"])

    assert runs[0] == runs[1] == expected


# ── transcribe ──────────────────────────────────────────────────────


def test_transcribe_selection_filters_to_split(tmp_path: Path) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    audio_dir = tmp_path / "audio_in"
    for stem in ("train_a", "test_a", "test_b"):
        _write_wav(audio_dir / f"{stem}.wav")

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "transcribe",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--audio", str(audio_dir),
            "--output", str(out_dir),
        ],
    )
    assert result.exit_code == 0, f"transcribe failed:\n{result.output}"
    assert {path.stem for path in out_dir.rglob("*.mid")} == {"test_a", "test_b"}
    assert "selection: mini split=test -> 2/3 files" in result.stdout


def test_transcribe_limit_is_deterministic_by_default(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    audio_dir = tmp_path / "audio_in"
    for index in range(8):
        _copy_midi(f"piece_{index}", midi_dir)
        _write_wav(audio_dir / f"piece_{index}.wav")

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    expected = sample_units(
        sorted(audio_dir.glob("*.wav")), SelectionSample(n=3, seed=0)
    )
    expected_stems = {path.stem for path in expected}

    runs: list[set[str]] = []
    for index in range(2):
        out_dir = tmp_path / f"out_{index}"
        result = CliRunner().invoke(
            app,
            [
                "transcribe",
                "--config", str(config_path),
                "--audio", str(audio_dir),
                "--output", str(out_dir),
                "--limit", "3",
            ],
        )
        assert result.exit_code == 0, f"transcribe failed:\n{result.output}"
        runs.append({path.stem for path in out_dir.rglob("*.mid")})

    assert runs[0] == runs[1] == expected_stems


def test_transcribe_audio_without_dataset_or_selection_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sonitra.selection as selection_module

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "pair_audio_to_reference must not run without a where filter"
        )

    monkeypatch.setattr(selection_module, "pair_audio_to_reference", _explode)

    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    audio_dir = tmp_path / "audio_in"
    for index in range(2):
        _copy_midi(f"piece_{index}", midi_dir)
        _write_wav(audio_dir / f"piece_{index}.wav")

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    out_dir = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "transcribe",
            "--config", str(config_path),
            "--audio", str(audio_dir),
            "--output", str(out_dir),
            "--limit", "1",
        ],
    )
    assert result.exit_code == 0, f"transcribe failed:\n{result.output}"
    assert len(list(out_dir.rglob("*.mid"))) == 1


# ── evaluate ────────────────────────────────────────────────────────


def test_evaluate_selection_filters_to_split(tmp_path: Path) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    est_dir = tmp_path / "est"
    for stem in ("train_a", "test_a", "test_b"):
        _copy_midi(stem, est_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=est_dir,
        ),
    )
    output = tmp_path / "results.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--reference", str(midi_dir),
            "--estimate", str(est_dir),
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {Path(row["file"]).stem for row in rows} == {"test_a", "test_b"}
    assert "selection: mini split=test -> 2/3 files" in result.stdout


def test_evaluate_limit_is_deterministic_by_default(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    est_dir = tmp_path / "est"
    for index in range(8):
        _copy_midi(f"piece_{index}", midi_dir)
        _copy_midi(f"piece_{index}", est_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=est_dir),
    )
    expected = {
        path.stem
        for path in sample_units(
            sorted(midi_dir.glob("*.mid")), SelectionSample(n=3, seed=0)
        )
    }

    runs: list[set[str]] = []
    for index in range(2):
        output = tmp_path / f"results_{index}.jsonl"
        result = CliRunner().invoke(
            app,
            [
                "evaluate",
                "--config", str(config_path),
                "--reference", str(midi_dir),
                "--estimate", str(est_dir),
                "--output", str(output),
                "--limit", "3",
            ],
        )
        assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
        runs.append(
            {Path(json.loads(line)["file"]).stem for line in output.read_text().splitlines()}
        )

    assert runs[0] == runs[1] == expected


def test_evaluate_limit_samples_only_refs_with_estimates_after_filter(
    tmp_path: Path,
) -> None:
    rows = [
        {"midi_filename": "train_a.mid", "split": "train"},
        {"midi_filename": "test_a.mid", "split": "test"},
        {"midi_filename": "test_b.mid", "split": "test"},
        {"midi_filename": "test_c.mid", "split": "test"},
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    est_dir = tmp_path / "est"
    for stem in ("test_a", "test_b", "train_a"):
        _copy_midi(stem, est_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block(
                {"split": ["test"]}, sample={"n": 2, "seed": 0}
            ),
            midi_dir=est_dir,
        ),
    )
    output = tmp_path / "results.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--reference", str(midi_dir),
            "--estimate", str(est_dir),
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
    rows_out = [json.loads(line) for line in output.read_text().splitlines()]
    assert {Path(row["file"]).stem for row in rows_out} == {"test_a", "test_b"}


def test_evaluate_config_sample_prefilters_estimates(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    est_dir = tmp_path / "est"
    for index in range(4):
        _copy_midi(f"piece_{index}", midi_dir)
    for index in range(3):
        _copy_midi(f"piece_{index}", est_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection={"sample": {"n": 2, "seed": 0}},
            midi_dir=est_dir,
        ),
    )
    output = tmp_path / "results.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config", str(config_path),
            "--reference", str(midi_dir),
            "--estimate", str(est_dir),
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, f"evaluate failed:\n{result.output}"
    rows_out = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows_out) == 2
    assert all(
        Path(row["file"]).stem in {"piece_0", "piece_1", "piece_2"}
        for row in rows_out
    )


# ── benchmark ───────────────────────────────────────────────────────


def test_benchmark_selection_filters_to_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    captured: dict[str, Any] = {}
    _patch_run_benchmark(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"
    assert [path.stem for path in captured["midi_paths"]] == ["test_a", "test_b"]
    assert "selection: mini split=test -> 2/3 files" in result.stdout


def test_benchmark_limit_is_deterministic_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data-root"
    midi_dir = root / DATASET / "midi"
    for index in range(8):
        _copy_midi(f"piece_{index}", midi_dir)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    expected = sample_units(
        sorted(midi_dir.glob("*.mid")), SelectionSample(n=3, seed=0)
    )

    runs: list[list[Path]] = []
    for index in range(2):
        captured: dict[str, Any] = {}
        _patch_run_benchmark(monkeypatch, captured)
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "--config", str(config_path),
                "--corpus", str(midi_dir),
                "--workdir", str(tmp_path / f"work_{index}"),
                "--limit", "3",
            ],
        )
        assert result.exit_code == 0, f"benchmark failed:\n{result.output}"
        runs.append(captured["midi_paths"])

    assert runs[0] == runs[1] == expected


def test_benchmark_saved_config_records_sample_from_limit(tmp_path: Path) -> None:
    rows = [
        {"midi_filename": "test_a.mid", "split": "test"},
        {"midi_filename": "test_b.mid", "split": "test"},
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    work_dir = tmp_path / "work"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--workdir", str(work_dir),
            "--limit", "1",
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"
    saved = yaml.safe_load((work_dir / "config.yaml").read_text())
    assert "selection" not in saved
    assert saved["io"]["sample"]["n"] == 1
    assert saved["io"]["sample"]["seed"] == 0


def test_benchmark_audio_mode_selection_then_sample_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"midi_filename": f"test_{index}.mid", "split": "test"} for index in range(4)
    ] + [
        {"midi_filename": f"train_{index}.mid", "split": "train"} for index in range(4)
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    recordings_dir = root / DATASET / "recordings"
    for row in rows:
        _write_wav(recordings_dir / f"{Path(row['midi_filename']).stem}.wav")

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            input_type="audio",
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    captured: dict[str, Any] = {}
    _patch_run_benchmark(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--limit", "3",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    test_recordings = sorted(recordings_dir.glob("test_*.wav"))
    expected_audio = sample_units(
        test_recordings, SelectionSample(n=3, seed=0)
    )
    expected_midi = sorted(midi_dir / f"{path.stem}.mid" for path in expected_audio)
    assert captured["audio_paths"] == sorted(expected_audio)
    assert captured["midi_paths"] == expected_midi
    assert "selection: mini split=test sample n=3 seed=0 -> 3/8 files" in result.stdout


def test_benchmark_audio_mode_samples_only_paired_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"midi_filename": "alpha.mid", "split": "test"},
        {"midi_filename": "beta.mid", "split": "test"},
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    recordings_dir = root / DATASET / "recordings"
    _write_wav(recordings_dir / "alpha.wav")
    _write_wav(recordings_dir / "beta.wav")
    _write_wav(recordings_dir / "orphan.wav")

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, input_type="audio", midi_dir=midi_dir),
    )
    captured: dict[str, Any] = {}
    _patch_run_benchmark(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--limit", "2",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"
    assert [path.stem for path in captured["audio_paths"]] == ["alpha", "beta"]
    assert [path.stem for path in captured["midi_paths"]] == ["alpha", "beta"]
    assert (
        "warning: 1 audio file(s) did not pair to a reference and were excluded"
        in result.stderr
    )
    assert "orphan" in result.stderr


def test_benchmark_corpus_override_outside_dataset_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    override_dir = tmp_path / "override"
    for stem in ("train_a", "test_a", "test_b"):
        _copy_midi(stem, override_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    captured: dict[str, Any] = {}
    _patch_run_benchmark(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--corpus", str(override_dir),
            "--workdir", str(tmp_path / "work"),
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"
    assert [path.stem for path in captured["midi_paths"]] == ["test_a", "test_b"]
    assert all(path.parent == override_dir for path in captured["midi_paths"])


# ── messages ────────────────────────────────────────────────────────


def test_split_notice_printed_when_unset_on_split_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    captured: dict[str, Any] = {}
    _patch_run_pipeline(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        [
            "render",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"
    # The console word-wraps long lines, so assert on the unwrapped pieces.
    assert "note: mini metadata labels splits (meta.csv: split)" in result.stdout
    assert "no selection set, using all" in result.stdout
    assert "3 files" in result.stdout


def test_seed_without_limit_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir),
    )
    captured: dict[str, Any] = {}
    _patch_run_pipeline(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        [
            "render",
            "--config", str(config_path),
            "--corpus", str(midi_dir),
            "--output", str(tmp_path / "out"),
            "--seed", "5",
        ],
    )
    assert result.exit_code == 0, f"render failed:\n{result.output}"
    assert (
        "warning: --seed has no effect without --limit or io.sample"
        in result.stderr
    )


# ---------------------------------------------------------------------------
# benchmark resume interacts with the recorded sample
# ---------------------------------------------------------------------------


def test_resume_with_changed_limit_is_refused(tmp_path: Path) -> None:
    rows = [
        {"midi_filename": "test_a.mid", "split": "test"},
        {"midi_filename": "test_b.mid", "split": "test"},
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir, benchmark_resume=True),
    )
    work_dir = tmp_path / "work"

    first = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--workdir", str(work_dir),
            "--limit", "1",
            "--seed", "0",
        ],
    )
    assert first.exit_code == 0, f"first benchmark failed:\n{first.output}"

    # The sample is part of the fingerprint, so a changed seed cannot resume
    # into the same work dir.
    second = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--workdir", str(work_dir),
            "--limit", "1",
            "--seed", "1",
        ],
    )
    assert second.exit_code != 0
    assert "fingerprint mismatch" in str(second.exception)


# ---------------------------------------------------------------------------
# Selection provenance in summary.json
# ---------------------------------------------------------------------------


def _expected_files_sha256(stems: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(stems)).encode("utf-8")).hexdigest()


def test_cli_benchmark_summary_selection_counts_and_hash(tmp_path: Path) -> None:
    rows = [
        {"midi_filename": "train_a.mid", "split": "train"},
        {"midi_filename": "train_b.mid", "split": "train"},
        {"midi_filename": "train_c.mid", "split": "train"},
        {"midi_filename": "test_a.mid", "split": "test"},
        {"midi_filename": "test_b.mid", "split": "test"},
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    work_dir = tmp_path / "work"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--workdir", str(work_dir),
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    selection = json.loads((work_dir / "summary.json").read_text())["selection"]
    assert selection["configured"] is True
    assert selection["dataset"] == DATASET
    assert selection["unit"] == "reference_midi"
    assert selection["counts"] == {
        "discovered": 5,
        "unmatched": 0,
        "excluded_by_where": 3,
        "unpaired_audio": 0,
        "selected_before_sample": 2,
        "selected": 2,
    }
    assert selection["files_sha256"] == _expected_files_sha256(
        ["test_a.mid", "test_b.mid"]
    )


def test_cli_benchmark_with_corpus_override_hashes_relative_to_actual_corpus(
    tmp_path: Path,
) -> None:
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    override_dir = tmp_path / "override"
    for stem in ("train_a", "test_a", "test_b"):
        _copy_midi(stem, override_dir)

    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            selection=_selection_block({"split": ["test"]}),
            midi_dir=midi_dir,
        ),
    )
    work_dir = tmp_path / "work"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--corpus", str(override_dir),
            "--workdir", str(work_dir),
        ],
    )
    assert result.exit_code == 0, f"benchmark failed:\n{result.output}"

    selection = json.loads((work_dir / "summary.json").read_text())["selection"]
    assert selection["configured"] is True
    assert selection["metadata_csv"] == str(
        root / DATASET / "metadata" / "meta.csv"
    )
    assert selection["counts"]["selected"] == 2
    assert selection["files_sha256"] == _expected_files_sha256(
        ["test_a.mid", "test_b.mid"]
    )


def test_resumed_benchmark_rewrites_selection_block(tmp_path: Path) -> None:
    rows = [
        {"midi_filename": "test_a.mid", "split": "test"},
        {"midi_filename": "test_b.mid", "split": "test"},
    ]
    root, midi_dir = _make_corpus(tmp_path, rows)
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(root, midi_dir=midi_dir, benchmark_resume=True),
    )
    work_dir = tmp_path / "work"

    first = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--workdir", str(work_dir),
        ],
    )
    assert first.exit_code == 0, f"first benchmark failed:\n{first.output}"
    first_selection = json.loads((work_dir / "summary.json").read_text())["selection"]

    second = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config", str(config_path),
            "--dataset", DATASET,
            "--workdir", str(work_dir),
        ],
    )
    assert second.exit_code == 0, f"resume failed:\n{second.output}"
    second_selection = json.loads((work_dir / "summary.json").read_text())["selection"]

    assert second_selection == first_selection
    assert second_selection["counts"]["selected"] == 2


# ---------------------------------------------------------------------------
# --dataset versus a YAML that filters its own dataset
# ---------------------------------------------------------------------------

_COMMANDS = ("render", "transcribe", "evaluate", "benchmark")


def _dataset_flag_case(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    filtered: bool,
    dataset_flag: str,
) -> tuple[list[str], Path]:
    """Build a config and CLI args for *command*; also return its output path."""
    root, midi_dir = _make_corpus(tmp_path, _SPLIT_ROWS)
    est_dir = tmp_path / "est"
    for stem in ("train_a", "test_a", "test_b"):
        _copy_midi(stem, est_dir)
    audio_dir = tmp_path / "audio_in"
    for stem in ("train_a", "test_a", "test_b"):
        _write_wav(audio_dir / f"{stem}.wav")

    selection = _selection_block({"split": ["test"]}) if filtered else None
    config_path = _write_config(
        tmp_path / "config.yaml",
        _config_payload(
            root,
            dataset=DATASET,
            selection=selection,
            midi_dir=est_dir if command == "evaluate" else midi_dir,
        ),
    )
    args = [command, "--config", str(config_path), "--dataset", dataset_flag]
    if command == "render":
        _patch_run_pipeline(monkeypatch, {})
        output = tmp_path / "out"
        args += ["--output", str(output)]
    elif command == "transcribe":
        output = tmp_path / "out"
        args += ["--audio", str(audio_dir), "--output", str(output)]
    elif command == "evaluate":
        output = tmp_path / "results.jsonl"
        args += ["--reference", str(midi_dir), "--estimate", str(est_dir)]
        args += ["--output", str(output)]
    else:
        _patch_run_benchmark(monkeypatch, {})
        output = tmp_path / "work"
        args += ["--workdir", str(output)]
    return args, output


@pytest.mark.parametrize("command", _COMMANDS)
def test_dataset_flag_conflicting_with_filtered_yaml_exits_1(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output = _dataset_flag_case(
        command, tmp_path, monkeypatch, filtered=True, dataset_flag="other"
    )
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, f"{command} did not fail:\n{result.output}"
    # The console word-wraps long lines, so compare on collapsed whitespace.
    stderr = " ".join(result.stderr.split())
    assert (
        "error: this config filters dataset 'mini' (io.where); --dataset 'other' "
        "would apply that filter to another dataset"
    ) in stderr
    assert not output.exists()


@pytest.mark.parametrize("command", _COMMANDS)
def test_dataset_flag_equal_to_filtered_yaml_is_accepted(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _dataset_flag_case(
        command, tmp_path, monkeypatch, filtered=True, dataset_flag=DATASET
    )
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, f"{command} failed:\n{result.output}"
    assert "selection: mini split=test -> 2/3 files" in result.stdout


@pytest.mark.parametrize("command", _COMMANDS)
def test_dataset_flag_on_unfiltered_yaml_overrides(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A YAML dataset without a filter is overridden by --dataset, as before.
    args, _ = _dataset_flag_case(
        command, tmp_path, monkeypatch, filtered=False, dataset_flag="other"
    )
    result = CliRunner().invoke(app, args)
    assert "this config filters dataset" not in " ".join(result.stderr.split())


def test_apply_dataset_sets_flag_on_unfiltered_config(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(
        _config_payload(tmp_path / "data-root", dataset=DATASET, midi_dir=tmp_path)
    )
    _apply_dataset(cfg, "other")
    assert cfg.io.dataset == "other"


def test_apply_dataset_conflict_leaves_config_unchanged(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(
        _config_payload(
            tmp_path / "data-root",
            selection=_selection_block({"split": ["test"]}),
            midi_dir=tmp_path,
        )
    )
    with pytest.raises(SelectionError, match="this config filters dataset 'mini'"):
        _apply_dataset(cfg, "other")
    assert cfg.io.dataset == DATASET
    _apply_dataset(cfg, DATASET)
    _apply_dataset(cfg, None)
    assert cfg.io.dataset == DATASET
