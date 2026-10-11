from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from sonitra.config import load_config
from sonitra.evaluation.protocol import evaluate_notes, make_symbolic_metrics
from sonitra.evaluation.types import notes_from_dicts
from sonitra.midi_reader import parse_midi
from sonitra.pipeline import run_pipeline
from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

_SOUNDFONT = Path("/usr/share/sounds/sf2/default-GM.sf2")
_FIXTURES = Path(__file__).parent / "fixtures"
_VITAL_VST = Path("/workspace/plugin/vital/lib/vst3/Vital.vst3")


@pytest.mark.slow
def test_basic_pitch_roundtrip_c4_soundfont(config_fixture, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    if not _SOUNDFONT.exists():
        pytest.skip("soundfont not found")

    cfg = load_config(config_fixture("roundtrip/dawdreamer_soundfont.yaml"))
    cfg.render_pipeline.overwrite = True
    cfg.observability.write_manifest = False

    result = run_pipeline([_FIXTURES / "test_c4.mid"], out_dir=tmp_path / "audio", config=cfg)
    assert result.succeeded == 1, f"pipeline failed: {result.log}"
    assert result.failed == 0

    audio_files: List[Path] = list((tmp_path / "audio").glob("*.wav"))
    assert len(audio_files) == 1, f"expected 1 WAV, found {audio_files}"

    t = BasicPitchTranscriber()
    tr = t.transcribe(audio_files[0])
    assert len(tr.notes) >= 1, "transcription returned no notes"

    ref = notes_from_dicts(parse_midi(_FIXTURES / "test_c4.mid"))
    est = notes_from_dicts(tr.notes)
    metrics = make_symbolic_metrics(cfg.evaluation)
    scores = evaluate_notes(ref, est, metrics)

    onset_recall = scores.get("note.onset_recall", 0.0)
    assert not (onset_recall != onset_recall), "onset_recall is NaN"
    assert onset_recall >= 1.0, f"Expected onset_recall == 1.0 (C4 must be detected), got {onset_recall:.3f} (scores: {scores})"

    onset_f1 = scores.get("note.onset_f1", 0.0)
    assert onset_f1 > 0.15, f"Expected onset_f1 > 0.15, got {onset_f1:.3f} (scores: {scores})"


@pytest.mark.slow
def test_basic_pitch_roundtrip_polyphonic_soundfont(config_fixture, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    if not _SOUNDFONT.exists():
        pytest.skip("soundfont not found")

    cfg = load_config(config_fixture("roundtrip/dawdreamer_soundfont.yaml"))
    cfg.render_pipeline.overwrite = True
    cfg.observability.write_manifest = False

    result = run_pipeline([_FIXTURES / "test_polyphonic.mid"], out_dir=tmp_path / "audio", config=cfg)
    assert result.succeeded == 1, f"pipeline failed: {result.log}"
    assert result.failed == 0

    audio_files: List[Path] = list((tmp_path / "audio").glob("*.wav"))
    assert len(audio_files) == 1

    t = BasicPitchTranscriber()
    tr = t.transcribe(audio_files[0])

    ref = notes_from_dicts(parse_midi(_FIXTURES / "test_polyphonic.mid"))
    est = notes_from_dicts(tr.notes)
    metrics = make_symbolic_metrics(cfg.evaluation)
    scores = evaluate_notes(ref, est, metrics)

    onset_recall = scores.get("note.onset_recall", 0.0)
    assert not (onset_recall != onset_recall), "onset_recall is NaN"
    assert onset_recall > 0.5, f"Expected onset_recall > 0.5 for chord, got {onset_recall:.3f} (scores: {scores})"

    distinct_pitches = {n["pitch"] for n in tr.notes}
    assert len(distinct_pitches) >= 2, f"expected >= 2 distinct pitches, got {distinct_pitches}"


_SOUNDFONT_CONFIGS = [
    "roundtrip/faust.yaml",
    "roundtrip/dawdreamer_soundfont.yaml",
    "roundtrip/pedalboard_no_effects.yaml",
    "roundtrip/pedalboard_all_effects.yaml",
]


@pytest.mark.slow
@pytest.mark.parametrize("config_name", _SOUNDFONT_CONFIGS)
def test_roundtrip_soundfont_config(config_name: str, config_fixture, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    if config_name != "roundtrip/faust.yaml":
        if not _SOUNDFONT.exists():
            pytest.skip("soundfont not found")

    cfg = load_config(config_fixture(config_name))
    cfg.render_pipeline.overwrite = True
    cfg.observability.write_manifest = False

    result = run_pipeline(
        [_FIXTURES / "test_polyphonic.mid"], out_dir=tmp_path / "audio", config=cfg
    )
    assert result.succeeded == 1, f"pipeline failed: {result.log}"
    assert result.failed == 0

    audio_files: List[Path] = list((tmp_path / "audio").glob("*.wav"))
    assert len(audio_files) == 1
    assert audio_files[0].stat().st_size > 0

    t = BasicPitchTranscriber()
    tr = t.transcribe(audio_files[0])
    assert isinstance(tr.notes, list)

    ref = notes_from_dicts(parse_midi(_FIXTURES / "test_polyphonic.mid"))
    est = notes_from_dicts(tr.notes)
    metrics = make_symbolic_metrics(cfg.evaluation)
    scores = evaluate_notes(ref, est, metrics)

    print(f"{config_name}: onset_f1={scores.get('note.onset_f1', float('nan')):.3f}, notes_found={len(tr.notes)}")


_VITAL_CONFIGS = [
    "config_dawdreamer_vital.yaml",
    "roundtrip/dawdreamer_vital_pedalboard.yaml",
]


@pytest.mark.slow
@pytest.mark.skip_if_no_vst
@pytest.mark.parametrize("config_name", _VITAL_CONFIGS)
def test_roundtrip_vital_config(config_name: str, config_fixture, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    if not _VITAL_VST.exists():
        pytest.skip("Vital VST3 not found")

    cfg = load_config(config_fixture(config_name))
    cfg.render_pipeline.overwrite = True
    cfg.observability.write_manifest = False

    result = run_pipeline(
        [_FIXTURES / "test_polyphonic.mid"], out_dir=tmp_path / "audio", config=cfg
    )
    assert result.succeeded == 1, f"pipeline failed: {result.log}"
    assert result.failed == 0

    audio_files: List[Path] = list((tmp_path / "audio").glob("*.wav"))
    assert len(audio_files) == 1
    assert audio_files[0].stat().st_size > 0

    t = BasicPitchTranscriber()
    tr = t.transcribe(audio_files[0])
    assert isinstance(tr.notes, list)

    ref = notes_from_dicts(parse_midi(_FIXTURES / "test_polyphonic.mid"))
    est = notes_from_dicts(tr.notes)
    metrics = make_symbolic_metrics(cfg.evaluation)
    scores = evaluate_notes(ref, est, metrics)

    print(f"{config_name}: onset_f1={scores.get('note.onset_f1', float('nan')):.3f}, notes_found={len(tr.notes)}")
