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


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "skip_if_no_vst" in item.keywords:
        path = os.getenv("VST_PATH") or os.getenv("VST3_PATH")
        if not path:
            pytest.skip("VST plugin path not configured")
