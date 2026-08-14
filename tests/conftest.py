from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


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

    Recordings are named ``piece_<n>_<performer>.wav`` so they pair (§2.3's
    token-prefix rule) to their reference ``piece_<n>.mid``.
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
    stays at the 44100Hz default — the regression fixture for the sample-rate
    structural fix in Phase 3 (audio-mode output must follow the source
    file's own rate, not the config rate)."""
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
    if "skip_if_no_vst" in item.keywords:
        path = os.getenv("VST_PATH") or os.getenv("VST3_PATH")
        if not path:
            pytest.skip("VST plugin path not configured")
