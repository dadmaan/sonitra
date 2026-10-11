"""Contract tests for the standalone benchmark-example rendering script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from sonitra.midi_reader import parse_midi
from sonitra.midi_writer import write_midi, write_multi_program_midi
from sonitra.pipeline import PipelineResult
from sonitra.storage import read_audio, write_wav
from tests.helpers import assert_notes_satisfy_contract

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "render_benchmark_examples.py"
SMOKE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "config_benchmark_render.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SONDFONT = Path("/usr/share/sounds/sf2/default-GM.sf2")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_benchmark_examples", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


def _write_tone(path: Path, *, sample_rate: int, seconds: float, freq: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * freq * t)
    return write_wav(np.stack([signal, signal]), path, sample_rate=sample_rate, normalize=False)


def _make_corpus(root: Path) -> None:
    """Two datasets: ``paired`` has recordings, ``midi_only`` has none."""
    for dataset in ("paired", "midi_only"):
        midi_dir = root / dataset / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / "test_c4.mid", midi_dir / "piece_0.mid")
        shutil.copy(FIXTURES / "test_polyphonic.mid", midi_dir / "piece_1.mid")
        write_midi(
            [
                {"pitch": 55, "velocity": 90, "start_sec": 0.0, "duration_sec": 0.5},
                {"pitch": 62, "velocity": 80, "start_sec": 0.5, "duration_sec": 0.5},
            ],
            midi_dir / "piece_2.mid",
        )
    for index, freq in enumerate((440.0, 523.25, 329.63)):
        _write_tone(
            root / "paired" / "recordings" / f"piece_{index}_performer1.wav",
            sample_rate=22050,
            seconds=2.0,
            freq=freq,
        )


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _make_corpus(root)
    return root


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "configs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def write_config(corpus_root: Path, configs_dir: Path):
    """Write a benchmark config derived from the smoke config under *configs_dir*."""

    def _write(
        relative: str,
        *,
        dataset: str | None = "paired",
        input_type: str = "midi",
        where: dict[str, list[str]] | None = None,
        metadata_csv: str | None = None,
        include_baseline: bool = True,
        conditions: list[dict[str, Any]] | None = None,
        sweeps: list[dict[str, Any]] | None = None,
        output_format: str = "wav",
    ) -> Path:
        payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
        io = payload["io"]
        io["corpus_root"] = str(corpus_root)
        io["dataset"] = dataset
        io["output_format"] = output_format
        io["file_naming"] = "{stem}"
        io["where"] = where or {}
        io["metadata_csv"] = metadata_csv
        payload["render_pipeline"]["input_type"] = input_type
        payload["benchmark"] = {
            "include_baseline": include_baseline,
            "baseline_name": "baseline",
            "conditions": conditions or [],
            "sweeps": sweeps or [],
        }
        path = configs_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> list[dict[str, Any]]:
    """Replace the script's ``run_pipeline`` with a recorder that writes its output."""
    calls: list[dict[str, Any]] = []

    def _run(
        midi_paths,
        out_dir,
        *,
        config=None,
        corpus_root=None,
        **kwargs,
    ) -> PipelineResult:
        assert config is not None, "config must be forwarded to run_pipeline"
        sources = [Path(item) for item in midi_paths]
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        log: list[dict[str, Any]] = []
        for source in sources:
            name = config.io.file_naming.format(stem=source.stem)
            target = target_dir / f"{name}.{config.io.output_format}"
            write_wav(np.full((1, 441), 0.1, dtype=np.float32), target, sample_rate=8000)
            log.append(
                {
                    "midi": str(source),
                    "output": str(target),
                    "status": "succeeded",
                    "elapsed_seconds": 0.01,
                }
            )
        calls.append(
            {
                "sources": sources,
                "out_dir": target_dir,
                "config": config,
                "corpus_root": corpus_root,
                "kwargs": kwargs,
            }
        )
        return PipelineResult(
            succeeded=len(sources),
            failed=0,
            skipped=0,
            elapsed_seconds=0.01,
            log=log,
        )

    monkeypatch.setattr(script, "run_pipeline", _run)
    return calls


def _argv(output: Path, configs_dir: Path, *extra: str) -> list[str]:
    return [
        "--output",
        str(output),
        "--configs-dir",
        str(configs_dir),
        "--yes",
        *extra,
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(output: Path, stem: str) -> dict[str, Any]:
    return _read_json(output / stem / "manifest.json")


def _summary(output: Path) -> dict[str, Any]:
    return _read_json(output / "render_summary.json")


def _by_status(manifest: dict[str, Any]) -> dict[str, str]:
    return {entry["name"]: entry["status"] for entry in manifest["conditions"]}


def _patch_config(path: Path, values: dict[str, Any]) -> Path:
    """Set top-level config keys of an already written config, e.g. ``sweeps``."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for dotted, value in values.items():
        section, _, key = dotted.partition(".")
        payload[section][key] = value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _reports(out: Path) -> dict[str, dict[str, Any]]:
    return {Path(entry["config"]).stem: entry for entry in _summary(out)["configs"]}


# --- discovery and filtering -------------------------------------------------------


def test_discover_configs_scans_recursively_for_sorted_yaml_only(
    script: ModuleType, configs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (configs_dir / "a.yaml").write_text("io: {}\n")
    (configs_dir / "nested").mkdir()
    (configs_dir / "nested" / "b.yaml").write_text("io: {}\n")
    (configs_dir / "nested" / "deeper").mkdir()
    (configs_dir / "nested" / "deeper" / "c.yaml").write_text("io: {}\n")
    (configs_dir / "nested" / "skipped.yml").write_text("io: {}\n")
    (configs_dir / "README.md").write_text("not a config\n")

    absolute = script.discover_configs(configs_dir)

    assert [path.relative_to(configs_dir).as_posix() for path in absolute] == [
        "a.yaml",
        "nested/b.yaml",
        "nested/deeper/c.yaml",
    ]
    assert absolute == sorted(absolute)

    monkeypatch.chdir(configs_dir.parent)
    relative = script.discover_configs(Path(configs_dir.name))
    assert all(not path.is_absolute() for path in relative), relative
    assert [path.as_posix() for path in relative] == [
        "configs/a.yaml",
        "configs/nested/b.yaml",
        "configs/nested/deeper/c.yaml",
    ]


def test_filter_configs_matches_globs_relative_to_configs_dir(
    script: ModuleType, configs_dir: Path
) -> None:
    for relative in ("alpha.yaml", "beta/one.yaml", "beta/two.yaml", "gamma/three.yaml"):
        path = configs_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("io: {}\n")
    found = script.discover_configs(configs_dir)

    everything = script.filter_configs(found, [], [], configs_dir)
    assert [path.name for path in everything] == ["alpha.yaml", "one.yaml", "two.yaml", "three.yaml"]

    only_nested = script.filter_configs(found, ["beta/*"], [], configs_dir)
    assert [path.name for path in only_nested] == ["one.yaml", "two.yaml"]

    include_wins_none = script.filter_configs(found, None, [], configs_dir)
    assert include_wins_none == everything

    excluded = script.filter_configs(found, ["beta/*"], ["*one*"], configs_dir)
    assert [path.name for path in excluded] == ["two.yaml"]


# --- config selection ----------------------------------------------------------------


def test_duplicate_config_stem_is_a_usage_error(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("a/x.yaml")
    write_config("b/x.yaml")

    code = script.main(_argv(tmp_path / "out", configs_dir))

    assert code == 2
    assert rendered == []


# --- piece selection -----------------------------------------------------------------


def test_configs_on_one_dataset_share_the_sampled_piece(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
) -> None:
    write_config("one.yaml")
    write_config("two.yaml")
    write_config(
        "three.yaml",
        where={"split": ["test"]},
        metadata_csv="splits.csv",
    )

    first = tmp_path / "first"
    assert script.main(_argv(first, configs_dir, "--seed", "0")) == 0

    shared = {(_manifest(first, stem)["source"]["reference"]) for stem in ("one", "two", "three")}
    assert len(shared) == 1
    picked = shared.pop()
    assert Path(picked).name in {"piece_0.mid", "piece_1.mid", "piece_2.mid"}
    # io.where on one config must not narrow the pick
    assert Path(picked).parent == configs_dir.parent / "corpus" / "paired" / "midi"

    repeat = tmp_path / "repeat"
    assert script.main(_argv(repeat, configs_dir, "--seed", "0")) == 0
    assert _manifest(repeat, "one")["source"]["reference"] == picked

    other_seed = tmp_path / "other-seed"
    assert script.main(_argv(other_seed, configs_dir, "--seed", "1")) == 0
    assert _manifest(other_seed, "one")["source"]["reference"] != picked

    pinned = tmp_path / "pinned"
    assert script.main(_argv(pinned, configs_dir, "--source", "paired=piece_2")) == 0
    pinned_manifest = _manifest(pinned, "one")
    assert Path(pinned_manifest["source"]["reference"]).name == "piece_2.mid"
    assert pinned_manifest["source"]["pinned"] is True


# --- audio pairing -------------------------------------------------------------------


def test_audio_config_renders_with_a_paired_recording(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
) -> None:
    write_config("audio.yaml", input_type="audio")

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir, "--source", "paired=piece_1")) == 0

    manifest = _manifest(out, "audio")
    assert manifest["input_type"] == "audio"
    assert Path(manifest["source"]["reference"]).name == "piece_1.mid"
    assert Path(manifest["source"]["recording"]).name == "piece_1_performer1.wav"
    assert _summary(out)["configs"][0]["status"] == "rendered"
    assert len(rendered) == 1


def test_audio_config_without_recordings_is_skipped_while_midi_renders(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
) -> None:
    write_config("a_audio.yaml", dataset="midi_only", input_type="audio")
    write_config("b_midi.yaml", dataset="midi_only")

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    reports = _reports(out)
    assert reports["a_audio"]["status"] == "skipped"
    assert reports["a_audio"]["reason"] == "no_recording"
    assert reports["b_midi"]["status"] == "rendered"
    assert len(rendered) == 1
    assert len(list((out / "b_midi").glob("*.wav"))) == 1


def test_dataset_without_reference_midi_skips_its_configs_only(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    corpus_root: Path,
) -> None:
    (corpus_root / "bare").mkdir()  # a dataset folder with no midi/ at all
    write_config("a_bare.yaml", dataset="bare")
    write_config("b_paired.yaml")

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    reports = _reports(out)
    assert reports["a_bare"]["status"] == "skipped"
    assert reports["a_bare"]["reason"] == "no_source"
    assert reports["b_paired"]["status"] == "rendered"
    assert len(rendered) == 1


def test_corrupt_midi_skips_its_config_and_the_run_continues(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    corpus_root: Path,
) -> None:
    broken = corpus_root / "broken" / "midi"
    broken.mkdir(parents=True)
    (broken / "piece_0.mid").write_bytes(b"not a midi file")
    write_config("a_broken.yaml", dataset="broken")
    write_config("b_paired.yaml")

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    reports = _reports(out)
    assert reports["a_broken"]["status"] == "skipped"
    assert reports["a_broken"]["reason"] == "no_source"
    assert reports["b_paired"]["status"] == "rendered"
    assert len(rendered) == 1
    assert (out / "b_paired" / "baseline.wav").exists()


# --- dataset resolution --------------------------------------------------------------


def test_config_without_dataset_is_skipped_without_a_default(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("no_dataset.yaml", dataset=None)

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    (report,) = _summary(out)["configs"]
    assert report["status"] == "skipped"
    assert report["reason"] == "no_dataset"
    assert rendered == []


def test_default_dataset_fills_in_a_missing_dataset(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("no_dataset.yaml", dataset=None)

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir, "--default-dataset", "paired")) == 0

    (report,) = _summary(out)["configs"]
    assert report["status"] == "rendered"
    assert report["reason"] is None
    manifest = _manifest(out, "no_dataset")
    assert manifest["dataset"] == "paired"
    assert Path(manifest["source"]["reference"]).parent == configs_dir.parent / "corpus" / "paired" / "midi"
    assert len(rendered) == 1


# --- excerpt cutting -----------------------------------------------------------------


def test_midi_excerpt_clips_and_shifts_notes_into_the_window(
    script: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source.mid"
    write_midi(
        [
            {"pitch": 60, "velocity": 100, "start_sec": 0.0, "duration_sec": 1.0},
            {"pitch": 64, "velocity": 90, "start_sec": 0.5, "duration_sec": 1.0},
            {"pitch": 67, "velocity": 80, "start_sec": 2.0, "duration_sec": 1.0},
        ],
        source,
        program=24,
    )
    piece = script.Piece(dataset="d", reference=source, recording=None, pinned=False, seed=0)

    excerpt = script.cut_excerpt(piece, "midi", 0.5, 1.0, tmp_path / "work")
    meta = parse_midi(excerpt, return_meta=True)

    assert meta["programs"] == [24]
    notes = [(n["pitch"], round(n["start_sec"], 3), round(n["duration_sec"], 3)) for n in meta["notes"]]
    assert notes == [(60, 0.0, 0.5), (64, 0.0, 1.0)]
    assert_notes_satisfy_contract(meta["notes"])
    assert excerpt != source

    assert script.cut_excerpt(piece, "midi", 0.0, None, tmp_path / "work") is source


def test_midi_excerpt_drops_the_program_when_the_source_has_several(
    script: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "multi.mid"
    write_multi_program_midi(
        [
            {"pitch": 60, "velocity": 90, "start_sec": 0.0, "duration_sec": 0.5, "program": 0},
            {"pitch": 64, "velocity": 90, "start_sec": 0.5, "duration_sec": 0.5, "program": 24},
        ],
        source,
    )
    piece = script.Piece(dataset="d", reference=source, recording=None, pinned=False, seed=0)

    excerpt = script.cut_excerpt(piece, "midi", 0.0, 1.0, tmp_path / "work")

    assert parse_midi(excerpt, return_meta=True)["programs"] == []


def test_audio_excerpt_slices_samples_at_the_native_rate(script: ModuleType, tmp_path: Path) -> None:
    recording = _write_tone(tmp_path / "take.wav", sample_rate=22050, seconds=4.0)
    piece = script.Piece(
        dataset="d", reference=tmp_path / "ref.mid", recording=recording, pinned=False, seed=0
    )

    excerpt = script.cut_excerpt(piece, "audio", 1.0, 2.0, tmp_path / "work")
    audio, sample_rate = read_audio(excerpt)

    assert sample_rate == 22050
    assert audio.shape[-1] == 2 * 22050
    assert excerpt != recording

    assert script.cut_excerpt(piece, "audio", 0.0, None, tmp_path / "work") is recording


def test_audio_excerpt_keeps_the_recording_level(script: ModuleType, tmp_path: Path) -> None:
    """The excerpt must not be rescaled: the effects chain sees the raw drive level."""
    recording = _write_tone(tmp_path / "take.wav", sample_rate=22050, seconds=4.0)
    piece = script.Piece(
        dataset="d", reference=tmp_path / "ref.mid", recording=recording, pinned=False, seed=0
    )

    excerpt = script.cut_excerpt(piece, "audio", 1.0, 2.0, tmp_path / "work")
    excerpt_audio, _ = read_audio(excerpt)
    source, _ = read_audio(recording)
    window = source[..., 22050 : 3 * 22050]

    assert float(np.max(np.abs(excerpt_audio))) == pytest.approx(
        float(np.max(np.abs(window))), abs=1e-3
    )
    assert 0.2 < float(np.max(np.abs(excerpt_audio))) < 0.4  # tone peak 0.3, not 1.0


# --- argument validation -------------------------------------------------------------


@pytest.mark.parametrize("value", ["10", "full"])
def test_duration_accepts_seconds_or_full(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    value: str,
) -> None:
    write_config("cfg.yaml")

    out = tmp_path / f"out-{value}"
    assert script.main(_argv(out, configs_dir, "--duration", value)) == 0
    assert _manifest(out, "cfg")["excerpt"]["duration_sec"] == (None if value == "full" else 10.0)


@pytest.mark.parametrize(
    "argv",
    [
        ["--duration", "0"],
        ["--duration", "-1"],
        ["--duration", "abc"],
        ["--duration", "10", "20"],
        ["--duration", "10", "--duration", "20"],
        ["--duration", "full", "--duration", "10"],
    ],
)
def test_duration_rejects_anything_else(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]],
    argv: list[str],
) -> None:
    write_config("cfg.yaml")

    code = script.main(_argv(tmp_path / "out", configs_dir, *argv))

    assert code == 2
    assert rendered == []


def test_output_pointing_at_a_file_is_a_usage_error(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("cfg.yaml")
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")

    code = script.main(_argv(blocker, configs_dir))

    assert code == 2
    assert rendered == []
    assert blocker.read_text(encoding="utf-8") == ""


def test_output_that_cannot_be_created_is_a_usage_error(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A parent that is a regular file passes every pre-flight check, then fails at mkdir."""
    write_config("cfg.yaml")
    blocker = tmp_path / "afile"
    blocker.write_text("", encoding="utf-8")
    out = blocker / "renders"

    code = script.main(_argv(out, configs_dir))

    assert code == 2, "no traceback may reach the user"
    captured = capsys.readouterr()
    assert "could not create" in captured.err + captured.out
    assert str(out) in captured.err + captured.out
    assert rendered == []
    assert not out.exists()
    assert blocker.read_text(encoding="utf-8") == ""


def test_start_past_the_end_of_the_piece_is_an_empty_window(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("cfg.yaml")

    out = tmp_path / "out"
    code = script.main(_argv(out, configs_dir, "--start", "100", "--duration", "5"))

    assert code == 0
    (report,) = _summary(out)["configs"]
    assert report["status"] == "skipped"
    assert report["reason"] == "empty_window"
    assert rendered == []


# --- per-condition rendering ---------------------------------------------------------


def test_condition_slug_names_the_output_file_with_equals_kept(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config(
        "cfg.yaml",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    assert (out / "cfg" / "wet_level=0.3.wav").exists()


def test_each_condition_gets_its_own_renaming_config_and_overrides(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config(
        "cfg.yaml",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    assert len(rendered) == 1
    call = rendered[0]
    assert len(call["sources"]) == 1
    assert call["corpus_root"] is None
    config = call["config"]
    assert config.io.file_naming == "wet_level=0.3"
    assert config.observability.write_manifest is False
    assert config.pedalboard.effects[1].wet_level == pytest.approx(0.3)


def test_bad_override_path_fails_only_its_own_condition(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config(
        "cfg.yaml",
        include_baseline=False,
        conditions=[
            {"name": "bad_key", "overrides": {"pipeline.synth_backend": "fluidsynth"}},
            {"name": "good", "overrides": {"pedalboard.effects.1.wet_level": 0.4}},
        ],
    )

    out = tmp_path / "out"
    code = script.main(_argv(out, configs_dir))

    assert code == 1
    conditions = {entry["name"]: entry for entry in _manifest(out, "cfg")["conditions"]}
    assert conditions["bad_key"]["status"] == "failed"
    assert "pipeline" in conditions["bad_key"]["error"]
    assert conditions["good"]["status"] == "succeeded"
    assert len(rendered) == 1


def test_existing_output_is_skipped_until_overwrite(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config(
        "cfg.yaml",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0
    assert len(rendered) == 1

    assert script.main(_argv(out, configs_dir)) == 0
    assert len(rendered) == 1, "pre-existing output must not be rendered again"
    (condition,) = _manifest(out, "cfg")["conditions"]
    assert condition["status"] == "skipped_existing"
    assert _summary(out)["configs"][0]["skipped_existing"] == 1

    assert script.main(_argv(out, configs_dir, "--overwrite")) == 0
    assert len(rendered) == 2
    assert _by_status(_manifest(out, "cfg"))["wet_level=0.3"] == "succeeded"


def test_pipeline_skipped_is_reported_as_skipped_existing(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file the pipeline itself skipped is not a failed condition."""
    write_config(
        "cfg.yaml",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    def _run(
        midi_paths,
        out_dir,
        *,
        config=None,
        corpus_root=None,
        **kwargs,
    ) -> PipelineResult:
        return PipelineResult(
            succeeded=0,
            failed=0,
            skipped=len(list(midi_paths)),
            elapsed_seconds=0.0,
            log=[
                {
                    "midi": str(midi_paths[0]),
                    "output": "",
                    "status": "skipped",
                    "elapsed_seconds": 0.0,
                }
            ],
        )

    monkeypatch.setattr(script, "run_pipeline", _run)

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir)) == 0

    (condition,) = _manifest(out, "cfg")["conditions"]
    assert condition["status"] == "skipped_existing"
    assert condition["error"] is None
    report = _summary(out)["configs"][0]
    assert report["status"] == "rendered"
    assert report["skipped_existing"] == 1


def test_overwrite_beats_a_config_that_disables_pipeline_overwrite(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    """``_render_file`` honours render_pipeline.overwrite, so --overwrite must override it."""
    path = write_config(
        "cfg.yaml",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )
    _patch_config(path, {"render_pipeline.overwrite": False})

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir, "--overwrite")) == 0

    assert len(rendered) == 1
    assert rendered[0]["config"].render_pipeline.overwrite is True


# --- written artefacts ---------------------------------------------------------------


def test_manifest_config_yaml_and_summary_carry_the_documented_keys(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config(
        "cfg.yaml",
        where={"split": ["test"]},
        metadata_csv="splits.csv",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    out = tmp_path / "out"
    assert script.main(_argv(out, configs_dir, "--duration", "10")) == 0

    manifest = _manifest(out, "cfg")
    assert set(manifest) == {
        "config",
        "dataset",
        "dataset_source",
        "input_type",
        "source",
        "filters_ignored",
        "excerpt",
        "conditions",
    }
    assert manifest["dataset"] == "paired"
    assert manifest["dataset_source"] == "io.dataset"
    assert manifest["input_type"] == "midi"
    assert set(manifest["source"]) == {"reference", "recording", "pinned", "seed"}
    assert manifest["filters_ignored"] == {"where": {"split": ["test"]}, "sample": None}
    assert manifest["excerpt"] == {"start_sec": 0.0, "duration_sec": 10.0}
    (condition,) = manifest["conditions"]
    assert set(condition) == {
        "name",
        "slug",
        "overrides",
        "status",
        "output",
        "error",
        "render_seconds",
    }
    assert condition["name"] == "wet_level=0.3"
    assert condition["slug"] == "wet_level=0.3"
    assert condition["overrides"] == {"pedalboard.effects.1.wet_level": 0.3}
    assert condition["output"] == "wet_level=0.3.wav"
    assert condition["error"] is None
    assert condition["render_seconds"] >= 0.0

    dumped = yaml.safe_load((out / "cfg" / "config.yaml").read_text(encoding="utf-8"))
    assert dumped["io"]["dataset"] == "paired"
    assert dumped["io"]["corpus_root"] == str(configs_dir.parent / "corpus")
    assert dumped["benchmark"]["include_baseline"] is False

    summary = _summary(out)
    assert set(summary) == {"created", "args", "pieces", "configs"}
    assert summary["pieces"]["paired"]["reference"] == manifest["source"]["reference"]
    assert summary["pieces"]["paired"]["pinned"] is False
    (report,) = summary["configs"]
    assert set(report) == {
        "config",
        "status",
        "reason",
        "succeeded",
        "failed",
        "skipped_existing",
        "out_dir",
    }
    assert report["status"] == "rendered"
    assert report["succeeded"] == 1
    assert report["failed"] == 0
    assert Path(report["out_dir"]) == out / "cfg"


@pytest.mark.parametrize("flag", ["--dry-run", "--list"])
def test_dry_run_and_list_write_nothing(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    flag: str,
) -> None:
    write_config("cfg.yaml")

    out = tmp_path / "out"
    code = script.main([*_argv(out, configs_dir), flag])

    assert code == 0
    assert not out.exists()
    assert rendered == []


def test_output_may_not_land_in_the_configs_dir_or_the_corpus(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("cfg.yaml")

    inside_configs = configs_dir / "renders"
    assert script.main(_argv(inside_configs, configs_dir)) == 2

    inside_corpus = configs_dir.parent / "corpus" / "paired" / "midi" / "renders"
    assert script.main(_argv(inside_corpus, configs_dir)) == 2

    assert rendered == []


def test_output_may_not_wrap_the_configs_dir_or_the_corpus(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    corpus_root: Path,
) -> None:
    """An output that contains an input would drop artefacts all over the corpus."""
    write_config("cfg.yaml")

    assert script.main(_argv(corpus_root, configs_dir)) == 2
    assert script.main(_argv(corpus_root / "paired", configs_dir)) == 2
    assert script.main(_argv(corpus_root / "paired" / "recordings", configs_dir)) == 2
    assert script.main(_argv(tmp_path, configs_dir)) == 2

    assert rendered == []
    assert not list(corpus_root.rglob("render_summary.json"))


def test_unmatched_include_and_a_missing_configs_dir_are_usage_errors(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config, rendered: list[dict[str, Any]]
) -> None:
    write_config("cfg.yaml")

    assert script.main(_argv(tmp_path / "out", tmp_path / "no-such-dir")) == 2
    assert script.main(_argv(tmp_path / "out", configs_dir, "--include", "no-such-*")) == 2
    assert script.main(_argv(tmp_path / "out", configs_dir, "--exclude", "*")) == 2

    assert rendered == []
    assert not (tmp_path / "out").exists()


def test_non_interactive_run_without_yes_is_a_usage_error(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_config("cfg.yaml")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    code = script.main(
        ["--output", str(tmp_path / "out"), "--configs-dir", str(configs_dir)]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "--yes" in captured.err + captured.out
    assert rendered == []


# --- real pipeline -------------------------------------------------------------------


@pytest.mark.skipif(not SONDFONT.exists(), reason="General MIDI SoundFont not available")
def test_real_pipeline_renders_two_condition_excerpts(
    script: ModuleType, tmp_path: Path, configs_dir: Path, write_config
) -> None:
    write_config(
        "cfg.yaml",
        include_baseline=True,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    out = tmp_path / "out"
    code = script.main(_argv(out, configs_dir, "--duration", "2", "--source", "paired=piece_1"))

    assert code == 0
    for name in ("baseline.wav", "wet_level=0.3.wav"):
        audio, sample_rate = read_audio(out / "cfg" / name)
        assert sample_rate == 44100
        assert float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))) > 0.001


def test_silent_excerpt_reports_the_quality_gate_that_failed(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    corpus_root: Path,
) -> None:
    """A quality-gate rejection carries its reason in quality_flags, not in error."""
    write_wav(
        np.zeros((1, 22050 * 2), dtype=np.float32),
        corpus_root / "paired" / "recordings" / "piece_0_performer1.wav",
        sample_rate=22050,
    )
    write_config(
        "cfg.yaml",
        input_type="audio",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.3]}],
    )

    out = tmp_path / "out"
    code = script.main(_argv(out, configs_dir, "--duration", "2", "--source", "paired=piece_0"))

    assert code == 1
    (condition,) = _manifest(out, "cfg")["conditions"]
    assert condition["status"] == "failed"
    assert "is_silent" in condition["error"]
    assert condition["error"] != "render reported status 'failed'"


# --- interactive toggle picker --------------------------------------------------------

_PICKER_CONFIGS = (
    "top.yaml",
    "methods/a.yaml",
    "methods/b.yaml",
    "nested/deeper/c.yaml",
    "sweeps/d.yaml",
    "sweeps/e.yaml",
)


def _chosen(configs_dir: Path, *relative: str) -> frozenset[Path]:
    """A picker selection, from paths written relative to *configs_dir*."""
    return frozenset(configs_dir / item for item in relative)


def _fake_input(
    monkeypatch: pytest.MonkeyPatch, script: ModuleType, answers: list[str]
) -> list[str]:
    """Answer the picker from *answers* through its one read function, recording prompts."""
    queue = list(answers)
    prompts: list[str] = []

    def _read(prompt: str) -> str:
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"the picker asked once more: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr(script, "_read_line", _read)
    return prompts


def _tokens(script: ModuleType, text: str) -> list[tuple[str, int, int, str]]:
    """A parsed line as comparable ``(kind, first, last, name)`` tuples."""
    return [(t.kind, t.first, t.last, t.name) for t in script.parse_tokens(text)]


@pytest.fixture
def picker(script: ModuleType, configs_dir: Path, write_config) -> Any:
    """A nine-row tree: a top-level config, three folders, one config nested deeper."""
    for relative in _PICKER_CONFIGS:
        write_config(relative)
    paths = [configs_dir / relative for relative in _PICKER_CONFIGS]
    targets = script.load_targets(paths, None)
    return paths, script.build_tree(paths, targets, configs_dir)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", [("submit", 0, 0, "")]),
        ("   ", [("submit", 0, 0, "")]),
        ("q", [("quit", 0, 0, "")]),
        ("Q", [("quit", 0, 0, "")]),
        ("a", [("all", 0, 0, "")]),
        ("n", [("none", 0, 0, "")]),
        ("i", [("invert", 0, 0, "")]),
        ("3", [("row", 3, 0, "")]),
        ("5-7", [("range", 5, 7, "")]),
        ("sweeps", [("folder", 0, 0, "sweeps")]),
        ("Methods/", [("folder", 0, 0, "Methods/")]),
        ("n sweeps 4", [("none", 0, 0, ""), ("folder", 0, 0, "sweeps"), ("row", 4, 0, "")]),
    ],
)
def test_parse_tokens_reads_every_documented_word(
    script: ModuleType, text: str, expected: list[tuple[str, int, int, str]]
) -> None:
    assert _tokens(script, text) == expected


@pytest.mark.parametrize("text", ["x-y", "5-", "-3", "3..5", "0", "7-3", "2 3..5", "5-a"])
def test_parse_tokens_rejects_malformed_input(script: ModuleType, text: str) -> None:
    with pytest.raises(script.ToggleError):
        script.parse_tokens(text)


def test_apply_toggle_numbers_and_ranges_touch_exactly_those_rows(
    script: ModuleType, configs_dir: Path, picker: Any
) -> None:
    _paths, tree = picker
    assert len(tree.rows) == 9, "the range in this test only means something with enough rows"
    everything = frozenset(tree.configs)

    toggled = script.apply_toggle(everything, script.parse_tokens("3 5-7"), tree)

    # row 3 clears methods/a, row 5 is the nested/ folder so it clears its only
    # config, row 6 puts that same config back and row 7 clears both sweeps configs
    assert toggled == _chosen(
        configs_dir, "top.yaml", "methods/b.yaml", "nested/deeper/c.yaml"
    )
    assert toggled < everything


def test_apply_toggle_rejects_an_unknown_row_or_folder_without_touching_the_state(
    script: ModuleType, configs_dir: Path, picker: Any
) -> None:
    _paths, tree = picker
    state = _chosen(configs_dir, "top.yaml", "sweeps/d.yaml")
    before = frozenset(state)

    with pytest.raises(script.ToggleError):
        script.apply_toggle(state, script.parse_tokens("99"), tree)
    with pytest.raises(script.ToggleError):
        script.apply_toggle(state, script.parse_tokens("nope"), tree)
    with pytest.raises(script.ToggleError):
        # the valid word before the bad one must not survive it
        script.apply_toggle(state, script.parse_tokens("1 99"), tree)

    assert state == before


def test_folder_rows_toggle_every_child_by_number_and_by_name(
    script: ModuleType, configs_dir: Path, picker: Any
) -> None:
    _paths, tree = picker
    everything = frozenset(tree.configs)

    by_number = script.apply_toggle(everything, script.parse_tokens("2"), tree)
    by_name = script.apply_toggle(everything, script.parse_tokens("methods"), tree)

    expected = _chosen(
        configs_dir, "top.yaml", "nested/deeper/c.yaml", "sweeps/d.yaml", "sweeps/e.yaml"
    )
    assert by_number == expected
    assert by_name == expected
    assert script.apply_toggle(everything, script.parse_tokens("Methods/"), tree) == expected

    partial = frozenset(path for path in tree.configs if path.name != "b.yaml")
    assert partial != everything
    assert script.apply_toggle(partial, script.parse_tokens("2"), tree) == everything


def test_a_n_and_i_set_the_whole_selection(script: ModuleType, configs_dir: Path, picker: Any) -> None:
    _paths, tree = picker
    everything = frozenset(tree.configs)
    nothing: frozenset[Path] = frozenset()

    assert script.apply_toggle(nothing, script.parse_tokens("a"), tree) == everything
    assert script.apply_toggle(everything, script.parse_tokens("n"), tree) == nothing
    assert script.apply_toggle(everything, script.parse_tokens("i"), tree) == nothing
    assert script.apply_toggle(nothing, script.parse_tokens("i"), tree) == everything

    half = _chosen(configs_dir, "top.yaml", "sweeps/d.yaml")
    assert script.apply_toggle(half, script.parse_tokens("i"), tree) == (everything - half)


def test_row_mark_reads_all_none_and_partial(script: ModuleType, configs_dir: Path, picker: Any) -> None:
    _paths, tree = picker
    folder_row = tree.rows[1]
    assert folder_row.is_folder

    assert script.row_mark(frozenset(tree.configs), (configs_dir / "top.yaml",)) == "[x]"
    assert script.row_mark(frozenset(), (configs_dir / "top.yaml",)) == "[ ]"
    assert script.row_mark(frozenset(tree.configs), folder_row.configs) == "[x]"
    assert script.row_mark(frozenset(), folder_row.configs) == "[ ]"
    assert (
        script.row_mark(_chosen(configs_dir, "methods/a.yaml"), folder_row.configs) == "[-]"
    )


def test_folder_rows_follow_the_paths_rather_than_a_fixed_depth(
    script: ModuleType, configs_dir: Path, write_config
) -> None:
    write_config("nested/deeper/c.yaml")
    write_config("top.yaml")

    paths = script.discover_configs(configs_dir)
    tree = script.build_tree(paths, script.load_targets(paths, None), configs_dir)

    assert [
        (row.number, row.label, row.is_folder, [path.name for path in row.configs])
        for row in tree.rows
    ] == [
        (1, "nested/", True, ["c.yaml"]),
        (2, "c.yaml", False, ["c.yaml"]),
        (3, "top.yaml", False, ["top.yaml"]),
    ]
    assert tree.configs == (configs_dir / "nested/deeper/c.yaml", configs_dir / "top.yaml")


def test_the_tree_names_folders_configs_and_counts_the_selection(
    script: ModuleType, configs_dir: Path, write_config, capsys: pytest.CaptureFixture[str]
) -> None:
    write_config("top.yaml")
    write_config("sweeps/a.yaml", input_type="audio")
    write_config("sweeps/b.yaml", input_type="audio", dataset=None)
    write_config(
        "sweeps/c.yaml",
        include_baseline=False,
        sweeps=[{"parameter": "pedalboard.effects.1.wet_level", "values": [0.1, 0.2]}],
    )

    paths = script.discover_configs(configs_dir)
    tree = script.build_tree(paths, script.load_targets(paths, None), configs_dir)
    # a wide console keeps every assertion on a single line
    script.print_tree(tree, frozenset(tree.configs), script.Console(width=200))

    printed = capsys.readouterr().out
    rows = [line for line in printed.splitlines() if line.lstrip().startswith("[")]
    assert len(rows) == 5
    assert f"Benchmark configs in {configs_dir}" in printed
    # the dataset-less audio config carries no conditions, so it adds none
    assert "selected 4/4 · 4 conditions" in printed
    assert "sweeps/" in rows[0] and "(3 configs, 3 conditions)" in rows[0]
    assert "a.yaml" in rows[1] and "audio · paired" in rows[1]
    assert "b.yaml" in rows[2] and "audio · -" in rows[2]
    assert "c.yaml" in rows[3] and " 2 " in rows[3], "the condition count is right-aligned"
    assert "top.yaml" in rows[4] and "audio ·" not in rows[4], "a midi config carries no note"
    assert (
        "Toggle: numbers / ranges (3 5-9) · folder name (sweeps) · a=all · "
        "n=none · i=invert · Enter=run · q=quit"
    ) in printed

    partial = _chosen(configs_dir, "top.yaml", "sweeps/a.yaml")
    script.print_tree(tree, partial, script.Console(width=200))
    printed = capsys.readouterr().out
    assert "[-]" in printed and "[ ]" in printed and "selected 2/4 · 2 conditions" in printed


def test_folders_and_configs_are_styled_apart(script: ModuleType, configs_dir: Path, write_config) -> None:
    """A folder must read as a group header, not as another config file."""
    import io

    write_config("sweeps/a.yaml")
    write_config("top.yaml")
    paths = script.discover_configs(configs_dir)
    tree = script.build_tree(paths, script.load_targets(paths, None), configs_dir)

    console = script.Console(file=io.StringIO(), force_terminal=True, width=200, record=True)
    script.print_tree(tree, frozenset(tree.configs), console)
    rendered = console.export_text(styles=True)

    def line_for(label: str) -> str:
        (line,) = [line for line in rendered.splitlines() if label in line]
        return line

    folder_line = line_for("sweeps/")
    config_line = line_for("top.yaml")
    # the folder carries the distinguishing style; the config does not
    assert folder_line != config_line
    assert str(tree.rows[0].label) == "sweeps/"
    assert _style_of(folder_line, "sweeps/") != _style_of(config_line, "top.yaml")
    # bold doubles the cue, so the distinction survives a colourless terminal
    assert "1" in _style_of(folder_line, "sweeps/"), "the folder style is not bold"


def _style_of(line: str, label: str) -> str:
    """The ANSI prefix in effect at *label* on a rendered line."""
    index = line.index(label)
    opening = line[:index]
    return opening[opening.rfind("\x1b[") :]


def test_prompt_loop_returns_the_folder_picked_after_a_bare_none(
    script: ModuleType,
    configs_dir: Path,
    picker: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _paths, tree = picker
    prompts = _fake_input(monkeypatch, script, ["n", "sweeps", ""])

    selection = script.prompt_tree(tree, script.Console(width=200))

    assert selection == _chosen(configs_dir, "sweeps/d.yaml", "sweeps/e.yaml")
    assert prompts == ["> "] * 3
    assert capsys.readouterr().out.count("Benchmark configs in") == 3


@pytest.mark.parametrize("answer", ["q", "Q", "q 3"])
def test_quitting_the_prompt_loop_returns_none(
    script: ModuleType, picker: Any, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    _paths, tree = picker
    _fake_input(monkeypatch, script, [answer])

    assert script.prompt_tree(tree, script.Console(width=200)) is None


def test_a_bad_word_re_prompts_without_changing_the_selection(
    script: ModuleType, configs_dir: Path, picker: Any, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _paths, tree = picker
    _fake_input(monkeypatch, script, ["99", "nope", "1", ""])

    selection = script.prompt_tree(tree, script.Console(width=200))

    assert selection == _chosen(
        configs_dir, "methods/a.yaml", "methods/b.yaml", "nested/deeper/c.yaml",
        "sweeps/d.yaml", "sweeps/e.yaml",
    )
    printed = capsys.readouterr().out
    assert "no row 99" in printed and "no folder named 'nope'" in printed


def test_enter_with_nothing_selected_asks_again(
    script: ModuleType,
    configs_dir: Path,
    picker: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _paths, tree = picker
    prompts = _fake_input(monkeypatch, script, ["n", "", "1", ""])

    selection = script.prompt_tree(tree, script.Console(width=200))

    assert selection == _chosen(configs_dir, "top.yaml")
    assert prompts == ["> "] * 4
    assert "nothing selected" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("Y", True), ("yes", True), ("YES", True), ("n", False), ("", False), ("yep", False)],
)
def test_only_yes_confirms_the_render(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected: bool,
) -> None:
    out = tmp_path / "out"
    prompts = _fake_input(monkeypatch, script, [answer])

    assert script._confirm_render(out, 7) is expected
    assert prompts == [f"Render 7 conditions into {out}? [y/N] "]


def test_dry_run_keeps_promising_that_nothing_is_written(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_config("cfg.yaml")

    out = tmp_path / "out"
    code = script.main(_argv(out, configs_dir, "--dry-run"))

    assert code == 0
    printed = capsys.readouterr().out
    assert "dry run: 1 config(s); nothing is written to" in printed
    assert not out.exists()


def test_interactive_run_renders_only_the_selected_configs(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_config("sweeps/a.yaml")
    write_config("sweeps/b.yaml")
    write_config("paper_experiments/c.yaml")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    prompts = _fake_input(monkeypatch, script, ["n", "sweeps", "", "y"])

    out = tmp_path / "out"
    code = script.main(["--output", str(out), "--configs-dir", str(configs_dir)])

    assert code == 0
    assert {Path(call["out_dir"]).name for call in rendered} == {"a", "b"}
    assert [entry["config"] for entry in _summary(out)["configs"]] == [
        str(configs_dir / "sweeps/a.yaml"),
        str(configs_dir / "sweeps/b.yaml"),
    ]
    assert (out / "a" / "manifest.json").exists()
    assert prompts[-1] == f"Render 2 conditions into {out}? [y/N] "
    # this run does write, so its plan header must not be the dry-run one
    assert "rendering plan: 2 config(s) into" in capsys.readouterr().out


def test_the_picker_starts_from_the_include_exclude_result(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config("sweeps/a.yaml")
    write_config("paper_experiments/c.yaml")
    write_config("paper_experiments/d.yaml")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    _fake_input(monkeypatch, script, ["", "y"])

    out = tmp_path / "out"
    code = script.main(
        ["--output", str(out), "--configs-dir", str(configs_dir), "--include", "sweeps/*"]
    )

    assert code == 0
    assert [Path(call["out_dir"]).name for call in rendered] == ["a"]


def test_quitting_the_picker_writes_nothing(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_config("cfg.yaml")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    _fake_input(monkeypatch, script, ["q"])

    out = tmp_path / "out"
    code = script.main(["--output", str(out), "--configs-dir", str(configs_dir)])

    assert code == 0
    assert not out.exists()
    assert rendered == []


@pytest.mark.parametrize("answer", ["", "n", "no"])
def test_declining_the_confirmation_writes_nothing(
    script: ModuleType,
    tmp_path: Path,
    configs_dir: Path,
    write_config,
    rendered: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    write_config("cfg.yaml")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    _fake_input(monkeypatch, script, ["n", "1", "", answer])

    out = tmp_path / "out"
    code = script.main(["--output", str(out), "--configs-dir", str(configs_dir)])

    assert code == 0
    assert not out.exists()
    assert rendered == []