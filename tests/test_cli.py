from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import pytest

from sonitra.cli import init
from sonitra.config import RenderingMode, load_config
from sonitra.pipeline import run_pipeline


def test_init_writes_working_basic_pitch_config(tmp_path: Path) -> None:
    path = tmp_path / "init.yaml"
    init(path)
    cfg = load_config(path)
    assert cfg.pipeline.rendering_mode == RenderingMode.DAWDREAMER_ONLY
    assert cfg.normalisation.enabled is True
    assert any(t.type == "basic_pitch" and t.enabled for t in cfg.transcription.transcribers)


def test_init_config_renders(corpus_dir: Path, tmp_path: Path) -> None:
    init(tmp_path / "init.yaml")
    cfg = load_config(tmp_path / "init.yaml")
    # Keep observability enabled but redirect manifest files into tmp_path so
    # tests do not write into the working directory.
    cfg.observability.manifest_path = tmp_path / "renders.jsonl"
    result = run_pipeline(sorted(corpus_dir.glob("*.mid")), tmp_path / "audio", config=cfg)
    assert result.succeeded >= 2


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_init_config_transcribes(corpus_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    from sonitra.transcribe.configs import BasicPitchTranscriberConfig
    from sonitra.transcribe.protocol import make_transcriber

    init(tmp_path / "init.yaml")
    cfg = load_config(tmp_path / "init.yaml")
    cfg.observability.manifest_path = tmp_path / "renders.jsonl"
    audio_dir = tmp_path / "audio"
    run_pipeline([corpus_dir / "test_c4.mid"], audio_dir, config=cfg)
    wav = next(audio_dir.glob("*.wav"))
    transcriber = make_transcriber(BasicPitchTranscriberConfig())
    result = transcriber.transcribe(wav)
    assert len(result.notes) > 0


def test_sonitra_console_script_is_registered() -> None:
    eps = entry_points(group="console_scripts")
    sonitra_eps = [ep for ep in eps if ep.name == "sonitra"]
    assert len(sonitra_eps) == 1
    assert sonitra_eps[0].value == "sonitra.cli:app"
