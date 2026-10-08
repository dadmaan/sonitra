from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from sonitra.storage import write_wav


def _transcribe_on_gpu(wav_path: str) -> int:
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    return len(BasicPitchTranscriber(device="GPU:0").transcribe(wav_path).notes)


@pytest.mark.slow
def test_gpu_transcription_works_in_spawned_worker_after_parent_preflight(
    tmp_path: Path,
) -> None:
    pytest.importorskip("basic_pitch")
    tf = pytest.importorskip("tensorflow")
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("no GPU visible to TensorFlow")

    # The parent runs the same checks as the benchmark preflight (device check
    # and numeric settings); spawned workers must still be able to use the GPU.
    BasicPitchTranscriber(device="GPU:0").validate_device()

    sample_rate = 22050
    t = np.linspace(0.0, 2.0, 2 * sample_rate, endpoint=False)
    wav = tmp_path / "tone.wav"
    write_wav(0.5 * np.sin(2 * np.pi * 440.0 * t), wav, sample_rate=sample_rate)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        assert pool.submit(_transcribe_on_gpu, str(wav)).result(timeout=300) >= 0


def _transcribe_and_numeric_fallbacks(wav_path: str) -> tuple[int, list[str] | None]:
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber

    result = BasicPitchTranscriber(
        device="GPU:0", numeric_mode="strict", gpu_memory_growth=True
    ).transcribe(wav_path)
    return len(result.notes), result.metadata.get("numeric_fallbacks")


@pytest.mark.slow
def test_gpu_worker_after_parent_numeric_preflight(tmp_path: Path) -> None:
    pytest.importorskip("basic_pitch")
    tf = pytest.importorskip("tensorflow")
    from sonitra.transcribe.basic_pitch import BasicPitchTranscriber
    from sonitra.transcribe.numerics import numeric_env

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("no GPU visible to TensorFlow")

    # The parent resolves the numeric settings exactly as the benchmark preflight
    # does, then leaves the device warm; a spawned worker must still get the same
    # settings with nothing left behind to break it.
    with numeric_env("strict", True):
        transcriber = BasicPitchTranscriber(device="GPU:0")
        transcriber.validate_device()
        transcriber.apply_numeric_settings()

    sample_rate = 22050
    t = np.linspace(0.0, 2.0, 2 * sample_rate, endpoint=False)
    wav = tmp_path / "tone.wav"
    write_wav(0.5 * np.sin(2 * np.pi * 440.0 * t), wav, sample_rate=sample_rate)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        notes, fallbacks = pool.submit(
            _transcribe_and_numeric_fallbacks, str(wav)
        ).result(timeout=300)

    assert notes >= 0
    assert fallbacks == []
