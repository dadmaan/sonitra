from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sonitra.config import IOSection, PipelineConfig, resolve_corpus_paths
from sonitra.corpus import (
    discover_audio_files,
    discover_midi_files,
    pair_audio_to_reference,
)


# ---------------------------------------------------------------------------
# discover_audio_files
# ---------------------------------------------------------------------------


def test_discover_audio_recursive_sorted_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.FLAC").write_bytes(b"")
    (tmp_path / "c.mp3").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "d.WAV").write_bytes(b"")
    # non-audio files must be ignored
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "reference.mid").write_bytes(b"")

    result = discover_audio_files(tmp_path)

    assert result == sorted(result)
    assert result == [
        tmp_path / "a.wav",
        tmp_path / "b.FLAC",
        tmp_path / "c.mp3",
        nested / "d.WAV",
    ]


def test_discover_audio_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "song.wav").mkdir()  # a directory named like an audio file
    result = discover_audio_files(tmp_path)
    assert result == []


def test_discover_audio_empty_dir(tmp_path: Path) -> None:
    assert discover_audio_files(tmp_path) == []


# ---------------------------------------------------------------------------
# discover_midi_files (sanity — full regression coverage lives in
# tests/test_corpus_discovery.py against the cli.py re-export)
# ---------------------------------------------------------------------------


def test_discover_midi_files_basic(tmp_path: Path) -> None:
    (tmp_path / "piece.mid").write_bytes(b"")
    (tmp_path / "piece2.MIDI").write_bytes(b"")
    (tmp_path / "audio.wav").write_bytes(b"")
    result = discover_midi_files(tmp_path)
    assert result == [tmp_path / "piece.mid", tmp_path / "piece2.MIDI"]


# ---------------------------------------------------------------------------
# pair_audio_to_reference
# ---------------------------------------------------------------------------


def _p(*names: str) -> list[Path]:
    return [Path(name) for name in names]


def test_pair_bsed_audio_to_midi() -> None:
    """§2.3 worked example: 3 recordings of the same excerpt all pair to the
    single reference MIDI that shares the excerpt-id token prefix."""
    audio_paths = _p(
        "BSED-01_1_Beethoven_Op021-01_Karajan1963.wav",
        "BSED-01_2_Beethoven_Op021-01_Ansermet1956.wav",
        "BSED-01_5_Beethoven_Op021-01_Synth.wav",
    )
    midi_paths = _p(
        "BSED-01_Beethoven_Op021-01.mid",
        "BSED-02_Mozart_K331.mid",
    )

    result = pair_audio_to_reference(audio_paths, midi_paths)

    reference = Path("BSED-01_Beethoven_Op021-01.mid")
    assert result.mapping == {
        Path("BSED-01_1_Beethoven_Op021-01_Karajan1963.wav"): reference,
        Path("BSED-01_2_Beethoven_Op021-01_Ansermet1956.wav"): reference,
        Path("BSED-01_5_Beethoven_Op021-01_Synth.wav"): reference,
    }
    assert result.unpaired_audio == []
    # BSED-02 has no recording -> reported as unpaired reference.
    assert result.unpaired_midi == [Path("BSED-02_Mozart_K331.mid")]


def test_pair_token_comparison_is_case_sensitive() -> None:
    audio_paths = _p("BSED-01_Beethoven.wav")
    midi_paths = _p("BSED-01_beethoven.mid", "BSED-01_other.mid")

    result = pair_audio_to_reference(audio_paths, midi_paths)

    # The case mismatch prevents the k=2 unique match; both references still
    # share the BSED-01 prefix at k=1, so the result is ambiguous, not paired.
    assert result.mapping == {}
    assert result.unpaired_audio == [Path("BSED-01_Beethoven.wav")]


def test_pair_descends_through_zero_match_levels() -> None:
    """Verbatim §2.3 worked example: k descends from len(A) through several
    zero-candidate levels (a longer decoy reference forces the start at
    k=5) before landing a unique match at k=1."""
    audio_paths = _p("BSED-01_1_Beethoven_Op021-01_Karajan1963.wav")
    midi_paths = _p(
        "BSED-01_Beethoven_Op021-01.mid",  # len(R)=3, the true match
        "BSED-02_Mozart_K331.mid",  # len(R)=3, decoy
        "BSED-99_Random_Decoy_Extra_Tokens.mid",  # len(R)=5, forces k_start=5
    )

    result = pair_audio_to_reference(audio_paths, midi_paths)

    assert result.mapping == {
        Path("BSED-01_1_Beethoven_Op021-01_Karajan1963.wav"): Path(
            "BSED-01_Beethoven_Op021-01.mid"
        )
    }


def test_pair_returns_unpaired_audio() -> None:
    audio_paths = _p("XYZ-99_Unknown_Piece.wav")
    midi_paths = _p("BSED-01_Beethoven_Op021-01.mid")

    result = pair_audio_to_reference(audio_paths, midi_paths)

    assert result.mapping == {}
    assert result.unpaired_audio == [Path("XYZ-99_Unknown_Piece.wav")]


def test_pair_returns_unpaired_midi() -> None:
    audio_paths = _p("BSED-01_1_Beethoven_Op021-01_Karajan1963.wav")
    midi_paths = _p(
        "BSED-01_Beethoven_Op021-01.mid",
        "BSED-02_Mozart_K331.mid",
    )

    result = pair_audio_to_reference(audio_paths, midi_paths)

    assert result.unpaired_midi == [Path("BSED-02_Mozart_K331.mid")]


def test_pair_ambiguous_match_stops_at_first_multi_candidate_k(
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio_paths = _p("BSED-07_X.wav")
    midi_paths = _p("BSED-07_X_extra1.mid", "BSED-07_X_extra2.mid")

    with caplog.at_level(logging.WARNING, logger="sonitra.corpus"):
        result = pair_audio_to_reference(audio_paths, midi_paths)

    assert result.mapping == {}
    assert result.unpaired_audio == [Path("BSED-07_X.wav")]
    # Exactly one ambiguity warning: proof the algorithm stopped at the first
    # multi-candidate k instead of continuing to smaller k values (which,
    # per the monotonic-superset property, could only ever grow the
    # candidate set further and would incorrectly log again).
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_pair_is_deterministic() -> None:
    audio_paths = _p(
        "BSED-01_1_Beethoven_Op021-01_Karajan1963.wav",
        "BSED-01_2_Beethoven_Op021-01_Ansermet1956.wav",
    )
    midi_paths = _p("BSED-01_Beethoven_Op021-01.mid", "BSED-02_Mozart_K331.mid")

    result1 = pair_audio_to_reference(audio_paths, midi_paths)
    result2 = pair_audio_to_reference(list(reversed(audio_paths)), midi_paths)

    assert result1.mapping == result2.mapping
    assert result1.unpaired_audio == result2.unpaired_audio
    assert result1.unpaired_midi == result2.unpaired_midi


# ---------------------------------------------------------------------------
# resolve_corpus_paths — recordings source dir
# ---------------------------------------------------------------------------


def _minimal_config_dict(*, dataset: str | None = None) -> dict:
    io: dict = {
        "corpus_root": "corpus",
        "output_format": "wav",
        "mp3_bitrate_kbps": 192,
        "file_naming": "{stem}",
    }
    if dataset is not None:
        io["dataset"] = dataset
    return {
        "render_pipeline": {
            "synth_backend": "dawdreamer_faust",
            "effects_chain": "none",
            "sample_rate": 44100,
            "bit_depth": 24,
            "channels": 2,
            "duration_padding_sec": 2.0,
            "overwrite": False,
            "resume": True,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": io,
    }


def test_resolve_corpus_paths_records_source_dir() -> None:
    cfg = PipelineConfig.model_validate(_minimal_config_dict(dataset="bsed"))
    paths = resolve_corpus_paths(cfg, config_name="pedalboard_baseline")

    assert paths.recordings == Path("corpus") / "bsed" / "recordings"
    # The output audio dir is unaffected by the new source attribute.
    assert paths.audio == Path("corpus") / "bsed" / "audio" / "pedalboard_baseline"
