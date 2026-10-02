"""Tests for the script measuring each recording's own tuning offset from A440."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import librosa
import mido
import numpy as np
import pytest
import soundfile

from sonitra.corpus import PairingResult, pair_audio_to_reference
from sonitra.effects.builtin_effects import TuningOffsetConfig
from sonitra.effects.chain_builder import TuningOffsetStage

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "estimate_tuning.py"
ENRICH_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "enrich_metadata.py"

# Fixture audio is written at 44.1 kHz and read back at the script's 22.05 kHz.
_FIXTURE_SAMPLE_RATE = 44100
_FIXTURE_DURATION_SEC = 4.0
_FIXTURE_NOTES: tuple[int, ...] = (45, 52, 57, 60, 64, 67, 72, 76, 81)
_FIXTURE_HARMONICS = 5
_LOAD_SAMPLE_RATE = 22050
_ONE_DECIMAL = re.compile(r"-?\d+\.\d")

# The estimator's own error is at most 2 cents, so a measured offset is pinned
# to a tolerance just above that, never to an exact number.
_CENTS_TOLERANCE = 3.0


def _absent_script(name: str, path: Path) -> ModuleType:
    """Stand-in for a missing script, so its tests fail rather than collection."""
    module = ModuleType(name)

    def main(argv: list[str] | None = None) -> int:
        raise AssertionError(f"{path} does not exist, so there is nothing to measure")

    module.main = main  # type: ignore[attr-defined]
    module.pair_audio_to_reference = pair_audio_to_reference  # type: ignore[attr-defined]
    return module


def _load_script(path: Path, name: str) -> ModuleType:
    """Execute the script at *path* under module name *name*."""
    if not path.exists():
        return _absent_script(name, path)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field types via sys.modules[cls.__module__], so a
    # module loaded by path has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def est() -> ModuleType:
    return _load_script(SCRIPT_PATH, "estimate_tuning")


@pytest.fixture()
def em() -> ModuleType:
    return _load_script(ENRICH_SCRIPT_PATH, "enrich_metadata")


# ── Fixture writers ──────────────────────────────────────────────────────


def _write_tuned_wav(path: Path, cents: float) -> Path:
    """Write a 4 s chord of nine harmonic notes tuned *cents* from A440.

    A chord rather than a lone note: the estimator bins energy over the whole
    spectrum, so a single partial is far too easy to estimate correctly by
    accident.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(int(_FIXTURE_SAMPLE_RATE * _FIXTURE_DURATION_SEC)) / _FIXTURE_SAMPLE_RATE
    scale = 2.0 ** (cents / 1200.0)
    mix = np.zeros_like(times)
    for midi in _FIXTURE_NOTES:
        fundamental = 440.0 * scale * 2.0 ** ((midi - 69) / 12.0)
        for harmonic in range(1, _FIXTURE_HARMONICS + 1):
            mix += np.sin(2.0 * np.pi * fundamental * harmonic * times) / harmonic
    mix = mix / float(np.max(np.abs(mix))) * 0.7
    soundfile.write(path, np.stack([mix, mix]).T, _FIXTURE_SAMPLE_RATE)
    return path


def _write_reference_midi(path: Path) -> Path:
    """Write a header-only MIDI file; reference pairing is extension-based."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mido.MidiFile().save(path)
    except Exception:
        path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x00")
    return path


def _write_corrupt_wav(path: Path) -> Path:
    """Write a file carrying a ``.wav`` ending but no decodable audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not audio, only a very misleading file name")
    return path


def _write_metadata(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> Path:
    """Write a dataset metadata CSV with exactly the given columns and rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── Output readers ───────────────────────────────────────────────────────


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _recordings_csv(output: Path) -> Path:
    return output.with_name(f"{output.stem}.recordings.csv")


def _row(rows: list[dict[str, str]], column: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row[column] == value]
    assert len(matches) == 1, f"expected exactly one {column}={value!r} row, got {matches}"
    return matches[0]


def _argv(recordings: Path, references: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--recordings",
        str(recordings),
        "--references",
        str(references),
        "--output",
        str(output),
        *extra,
    ]


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a provenance payload into dotted ``path -> scalar`` pairs."""
    flat: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = payload
    return flat


def _provenance_value(flat: dict[str, Any], needles: tuple[str, ...], types: Any) -> Any:
    """Return the provenance value whose key names *needles* and holds *types*.

    Different spellings of one fact are fine as long as they agree, which is
    why every match has to carry the same value.
    """
    matches = {
        key: value
        for key, value in flat.items()
        if any(needle in key for needle in needles) and isinstance(value, types)
    }
    assert matches, (
        f"no {types} provenance value under a key containing {needles}; "
        f"keys: {sorted(flat)}"
    )
    assert len({repr(value) for value in matches.values()}) == 1, (
        f"provenance values disagree for {needles}: {matches}"
    )
    return next(iter(matches.values()))


# ── Measurements ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("cents", [-30.0, -15.0, 0.0, 15.0, 30.0])
def test_estimates_known_offsets(est: ModuleType, tmp_path: Path, cents: float) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "riff.wav", cents)
    _write_reference_midi(references / "riff.mid")
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output)) == 0

    rows = _read_csv(_recordings_csv(output))
    assert [row["recording"] for row in rows] == ["riff.wav"]
    measured = rows[0]["own_tuning_cents"]
    assert _ONE_DECIMAL.fullmatch(measured), f"cents want one decimal, got {measured!r}"
    assert float(measured) == pytest.approx(cents, abs=_CENTS_TOLERANCE)

    reference_rows = _read_csv(output)
    assert [row["reference_stem"] for row in reference_rows] == ["riff"]
    assert reference_rows[0]["n_recordings"] == "1"
    assert float(reference_rows[0]["own_tuning_cents"]) == pytest.approx(
        cents, abs=_CENTS_TOLERANCE
    )


def test_estimates_tuning_offset_stage_output(est: ModuleType, tmp_path: Path) -> None:
    source = _write_tuned_wav(tmp_path / "source" / "riff.wav", 0.0)
    # Decoded at the rate and channel layout the script itself uses, so the
    # stage sees exactly the signal the script would have measured on disk.
    mono, _ = librosa.load(source, sr=_LOAD_SAMPLE_RATE, mono=True)
    stage = TuningOffsetStage(TuningOffsetConfig(cents=20.0))
    shifted = stage(mono.astype(np.float32)[None, :], _LOAD_SAMPLE_RATE)

    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    recordings.mkdir()
    soundfile.write(recordings / "shifted.wav", shifted.T, _LOAD_SAMPLE_RATE)
    _write_reference_midi(references / "shifted.mid")
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output)) == 0

    rows = _read_csv(_recordings_csv(output))
    assert float(rows[0]["own_tuning_cents"]) == pytest.approx(20.0, abs=_CENTS_TOLERANCE)


def test_pairs_recordings_with_corpus_pairing(
    est: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    paired_audio = _write_tuned_wav(recordings / "song.wav", 0.0)
    unpaired_audio = _write_tuned_wav(recordings / "solo.wav", 5.0)
    paired_midi = _write_reference_midi(references / "song.mid")
    calls: list[tuple[list[Path], list[Path]]] = []

    def spy(audio_paths: Sequence[Path], midi_paths: Sequence[Path]) -> PairingResult:
        calls.append((list(audio_paths), list(midi_paths)))
        return pair_audio_to_reference(audio_paths, midi_paths)

    monkeypatch.setattr(est, "pair_audio_to_reference", spy)
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output)) == 0

    assert len(calls) == 1, "pairing must happen once, over every discovered file"
    assert sorted(calls[0][0]) == sorted([paired_audio, unpaired_audio])
    assert sorted(calls[0][1]) == [paired_midi]

    reference_rows = _read_csv(output)
    assert [row["reference_stem"] for row in reference_rows] == [paired_midi.stem]

    rows = _read_csv(_recordings_csv(output))
    assert [row["recording"] for row in rows] == ["solo.wav", "song.wav"]
    assert _row(rows, "recording", "song.wav")["reference_stem"] == paired_midi.stem
    # An unpaired recording is still measured and still reported, but it has no
    # reference to contribute to, so it appears in no per-reference row.
    assert _row(rows, "recording", "solo.wav")["reference_stem"] == ""


def test_two_recordings_per_reference(est: ModuleType, tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "x_mic.wav", 10.0)
    _write_tuned_wav(recordings / "x_mix.wav", 14.0)
    _write_reference_midi(references / "x.mid")
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output)) == 0

    reference_rows = _read_csv(output)
    assert len(reference_rows) == 1, "one paired reference must yield one row"
    reference = reference_rows[0]
    assert reference["reference_stem"] == "x"
    assert reference["n_recordings"] == "2"
    assert float(reference["own_tuning_cents"]) == pytest.approx(12.0, abs=_CENTS_TOLERANCE)
    # Each of the two estimates carries up to 2 cents of its own quantisation
    # error, so the pair's spread reads 6.0 instead of the true 4.0; 3.0 of
    # tolerance still rejects a collapsed 0.0 spread and a one-recording value.
    assert float(reference["own_tuning_spread_cents"]) == pytest.approx(
        4.0, abs=_CENTS_TOLERANCE
    )

    rows = _read_csv(_recordings_csv(output))
    assert [row["recording"] for row in rows] == ["x_mic.wav", "x_mix.wav"]
    assert [row["reference_stem"] for row in rows] == ["x", "x"]
    assert [row["status"] for row in rows] == ["ok", "ok"]


def test_metadata_join_emits_exact_value(est: ModuleType, tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "name.wav", 0.0)
    _write_reference_midi(references / "2018" / "name.mid")

    metadata = _write_metadata(
        tmp_path / "meta.csv",
        ["midi_filename", "composer"],
        [{"midi_filename": "2018/name.midi", "composer": "Anon"}],
    )
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output, "--metadata", str(metadata))) == 0

    assert _read_header(output)[0] == "midi_filename", (
        "the join key has to come first, ready to be handed to a join tool"
    )
    row = _row(_read_csv(output), "midi_filename", "2018/name.midi")
    assert row["reference_stem"] == "name"

    # A GuitarSet-like table keys on a bare stem rather than a relative path,
    # so the join column has to be whatever the dataset happens to use.
    guitarset = _write_metadata(
        tmp_path / "guitarset.csv",
        ["id", "artist"],
        [{"id": "name", "artist": "Anon"}],
    )
    keyed = tmp_path / "tuning_by_id.csv"

    assert (
        est.main(
            _argv(
                recordings,
                references,
                keyed,
                "--metadata",
                str(guitarset),
                "--join-column",
                "id",
            )
        )
        == 0
    )

    assert _read_header(keyed)[0] == "id"
    assert _row(_read_csv(keyed), "id", "name")["reference_stem"] == "name"


def test_output_joins_with_enrich_metadata(
    est: ModuleType, em: ModuleType, tmp_path: Path
) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "name.wav", -15.0)
    _write_reference_midi(references / "2018" / "name.mid")
    metadata = _write_metadata(
        tmp_path / "meta.csv",
        ["midi_filename", "composer"],
        [
            {"midi_filename": "2018/name.midi", "composer": "Anon"},
            {"midi_filename": "2019/absent.midi", "composer": "Nobody"},
        ],
    )
    per_reference = tmp_path / "tuning.csv"

    assert (
        est.main(_argv(recordings, references, per_reference, "--metadata", str(metadata))) == 0
    )

    measured = {row["midi_filename"] for row in _read_csv(per_reference)}
    assert measured == {"2018/name.midi"}

    joined = tmp_path / "joined.csv"
    code = em.main(
        [
            "--metadata",
            str(metadata),
            "--annotations",
            str(per_reference),
            "--on",
            "midi_filename=midi_filename",
            "--add",
            "own_tuning_cents=own_tuning_cents",
            "--output",
            str(joined),
        ]
    )

    assert code == 0
    joined_rows = _read_csv(joined)
    assert len(joined_rows) == 2, "a left join must not drop metadata rows"
    assert _row(joined_rows, "midi_filename", "2018/name.midi")["own_tuning_cents"] != ""
    assert _row(joined_rows, "midi_filename", "2019/absent.midi")["own_tuning_cents"] == ""
    assert all(
        row["own_tuning_cents"] != "" for row in joined_rows if row["midi_filename"] in measured
    )


# ── Failure handling, guards and provenance ──────────────────────────────


def test_failed_row_with_comma_does_not_break_enrich(
    est: ModuleType, em: ModuleType, tmp_path: Path
) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "good.wav", 0.0)
    _write_corrupt_wav(recordings / 'bad, "quoted".wav')
    _write_reference_midi(references / "good.mid")
    metadata = _write_metadata(
        tmp_path / "meta.csv", ["midi_filename"], [{"midi_filename": "good.midi"}]
    )
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output, "--metadata", str(metadata))) == 0

    rows = _read_csv(_recordings_csv(output))
    assert len(rows) == 2
    assert _row(rows, "recording", "good.wav")["status"] == "ok"
    assert _row(rows, "recording", "bad; 'quoted'.wav")["status"] == "failed"
    assert len(_read_csv(output)) == 1, "one good recording is still a per-reference row"

    # Sanitised cells mean no field can ever need quoting, so a stray comma or
    # quote in a file name cannot shift every later column. Row separators are
    # the only newlines allowed to reach the file, so the raw fields are split
    # on commas rather than handed back with their quoting intact.
    text = _recordings_csv(output).read_bytes().decode("utf-8")
    assert '"' not in text, "no cell needs quoting, so no cell may be quoted"
    for line in text.splitlines():
        for cell in line.split(","):
            assert not any(char in cell for char in ',"\r\n'), f"raw cell {cell!r}"

    joined = tmp_path / "joined.csv"
    code = em.main(
        [
            "--metadata",
            str(metadata),
            "--annotations",
            str(output),
            "--on",
            "midi_filename=midi_filename",
            "--add",
            "own_tuning_cents=own_tuning_cents",
            "--output",
            str(joined),
        ]
    )

    assert code == 0


def test_fail_soft_on_unreadable_file(est: ModuleType, tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "ok.wav", 20.0)
    _write_corrupt_wav(recordings / "broken.wav")
    _write_reference_midi(references / "ok.mid")
    output = tmp_path / "tuning.csv"

    assert est.main(_argv(recordings, references, output)) == 0

    rows = _read_csv(_recordings_csv(output))
    assert [row["recording"] for row in rows] == ["broken.wav", "ok.wav"]

    failed = _row(rows, "recording", "broken.wav")
    assert failed["status"] == "failed"
    assert failed["error"] != "", "a failed row has to say why, or it is not diagnosable"

    measured = _row(rows, "recording", "ok.wav")
    assert measured["status"] == "ok"
    assert float(measured["own_tuning_cents"]) == pytest.approx(20.0, abs=_CENTS_TOLERANCE)
    assert float(measured["duration_sec"]) == pytest.approx(4.0, abs=0.05)

    # Only the recording that measured counts towards the reference, so one bad
    # file cannot drag a reference's statistics towards it.
    reference = _row(_read_csv(output), "reference_stem", "ok")
    assert reference["n_recordings"] == "1"


def test_refuses_to_overwrite_inputs(est: ModuleType, tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "ok.wav", 0.0)
    _write_reference_midi(references / "ok.mid")
    metadata = _write_metadata(
        tmp_path / "meta.csv", ["midi_filename"], [{"midi_filename": "ok.midi"}]
    )
    before = metadata.read_bytes()

    assert est.main(_argv(recordings, references, metadata, "--metadata", str(metadata))) != 0
    assert metadata.read_bytes() == before

    # Writing into the recordings tree would put a CSV in the next run's input.
    inside = recordings / "tuning.csv"
    assert est.main(_argv(recordings, references, inside, "--metadata", str(metadata))) != 0
    assert not inside.exists()


def test_provenance_written(est: ModuleType, tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "y_mic.wav", 10.0)
    _write_tuned_wav(recordings / "y_mix.wav", 14.0)
    _write_tuned_wav(recordings / "zz.wav", 0.0)
    _write_reference_midi(references / "y.mid")
    metadata = _write_metadata(
        tmp_path / "meta.csv", ["midi_filename"], [{"midi_filename": "y.midi"}]
    )
    output = tmp_path / "tuning.csv"
    argv = _argv(recordings, references, output, "--metadata", str(metadata))

    assert est.main(argv) == 0

    sidecar = Path(f"{output}.provenance.json")
    assert sidecar.exists()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    flat = _flatten(provenance)

    assert [str(item) for item in provenance["argv"]] == argv
    assert _provenance_value(flat, ("librosa",), str) == librosa.__version__
    assert (
        _provenance_value(flat, ("sonitra",), str)
        == importlib.metadata.version("sonitra")
    )
    assert _provenance_value(flat, ("sample_rate",), (int, float)) == _LOAD_SAMPLE_RATE
    assert _provenance_value(flat, ("resolution",), (int, float)) == 0.01
    assert _provenance_value(flat, ("bins_per_octave",), (int, float)) == 12
    assert _provenance_value(flat, ("chunk_seconds",), (int, float)) == 60.0
    assert isinstance(_provenance_value(flat, ("chunk",), bool), bool)

    assert _provenance_value(flat, ("measured",), int) == 3
    assert _provenance_value(flat, ("failed",), int) == 0
    assert _provenance_value(flat, ("unpaired",), int) == 1
    assert _provenance_value(flat, ("multi", "multiple", "several"), int) == 1
    assert (
        _provenance_value(flat, ("sha256",), str)
        == hashlib.sha256(metadata.read_bytes()).hexdigest()
    )


def test_empty_recordings_dir_errors(
    est: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    recordings = tmp_path / "corpus" / "recordings"
    recordings.mkdir(parents=True)
    references = tmp_path / "corpus" / "references"
    _write_reference_midi(references / "a.mid")
    output = tmp_path / "tuning.csv"

    code = est.main(_argv(recordings, references, output))

    assert code != 0
    assert not output.exists()
    captured = capsys.readouterr()
    message = captured.out + captured.err + caplog.text
    assert str(recordings) in message, f"the empty directory is not named in {message!r}"


def test_chunked_and_whole_file_estimates_agree(est: ModuleType, tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    references = tmp_path / "references"
    _write_tuned_wav(recordings / "chord.wav", 0.0)
    _write_reference_midi(references / "chord.mid")
    whole_file = tmp_path / "whole.csv"
    chunked = tmp_path / "chunked.csv"

    assert est.main(_argv(recordings, references, whole_file)) == 0
    assert est.main(_argv(recordings, references, chunked, "--chunk-seconds", "0.5")) == 0

    # librosa.pitch_tuning quantises to a whole cent, so agreeing inside half a
    # cent means both code paths resolved to the same cent.
    from_whole = float(_read_csv(_recordings_csv(whole_file))[0]["own_tuning_cents"])
    from_chunks = float(_read_csv(_recordings_csv(chunked))[0]["own_tuning_cents"])
    assert from_whole == pytest.approx(from_chunks, abs=0.5)
