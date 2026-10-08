"""Contract tests for the global tuning offset effect and the composite effects chain.

The offset goes through ``pedalboard.time_stretch`` because the ``PitchShift``
plugin does not apply cent-level shifts reliably. Signals are synthetic and
short, so the accuracy guard runs on every test run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard
import pytest
from pydantic import ValidationError

from sonitra.config import ConfigError, PipelineConfig
from sonitra.effects import EffectsChain
from sonitra.effects.builtin_effects import (
    CompressorConfig,
    DistortionConfig,
    GainConfig,
    LimiterConfig,
    ReverbConfig,
    TuningOffsetConfig,
)
from sonitra.effects.chain_builder import (
    CompositeEffectsChain,
    TuningOffsetStage,
    build_effects_chain,
    compute_chain_hash,
)
from sonitra.pipeline import run_pipeline
from sonitra.storage import read_audio, write_wav
from tests.test_pipeline_audio import _audio_config

SAMPLE_RATE = 44100
TONE_HZ = 220.0
WINDOW_SEC = 0.5
BAND_LO_HZ = 200.0
BAND_HI_HZ = 245.0

_REVERB = ReverbConfig(
    room_size=0.4,
    damping=0.5,
    wet_level=0.15,
    dry_level=0.85,
    width=1.0,
    freeze_mode=False,
)
_DISTORTION = DistortionConfig(drive_db=18.0)
_COMPRESSOR = CompressorConfig(
    threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=100.0
)


# ── Signal helpers ───────────────────────────────────────────────────


def _harmonic_tone(
    freq_hz: float = TONE_HZ,
    duration_sec: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
    harmonics: int = 5,
    peak: float = 0.5,
) -> np.ndarray:
    """Mono ``(1, n)`` harmonic tone; the fundamentals keep f0 estimation sharp."""
    t = np.arange(int(sample_rate * duration_sec)) / sample_rate
    signal = sum(
        np.sin(2.0 * np.pi * freq_hz * k * t) / k for k in range(1, harmonics + 1)
    )
    scaled = signal / float(np.max(np.abs(signal))) * peak
    return scaled.astype(np.float32)[None, :]


def _stereo_noise(samples: int = 4410, seed: int = 0) -> np.ndarray:
    """Seeded stereo noise, so bit-exact comparisons are reproducible."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, samples)) * 0.2).astype(np.float32)


def _estimate_f0(
    segment: np.ndarray,
    sample_rate: int,
    lo_hz: float = BAND_LO_HZ,
    hi_hz: float = BAND_HI_HZ,
) -> float:
    """Fundamental in ``[lo_hz, hi_hz]`` from a log-parabolic FFT peak."""
    flat = np.asarray(segment, dtype=np.float64)[0]
    n_fft = 1 << 18  # zero-pad for sub-bin peak placement
    spectrum = np.abs(np.fft.rfft(flat * np.hanning(flat.size), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    spectrum = spectrum * ((freqs >= lo_hz) & (freqs <= hi_hz))
    peak = int(np.argmax(spectrum))
    left, centre, right = np.log(spectrum[peak - 1 : peak + 2])
    offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
    return (peak + offset) * sample_rate / n_fft


def _measured_cents(
    audio: np.ndarray,
    sample_rate: int,
    reference_hz: float = TONE_HZ,
    window_sec: float = WINDOW_SEC,
    lo_hz: float = BAND_LO_HZ,
    hi_hz: float = BAND_HI_HZ,
) -> tuple[float, float]:
    """Mean and spread of the per-window offset from ``reference_hz``, in cents."""
    width = int(window_sec * sample_rate)
    offsets = [
        1200.0 * np.log2(_estimate_f0(audio[:, start : start + width], sample_rate, lo_hz, hi_hz) / reference_hz)
        for start in range(0, audio.shape[1] - width + 1, width)
    ]
    assert offsets, "signal is shorter than one estimation window"
    return float(np.mean(offsets)), float(np.std(offsets))


def _stage(cents: float, **overrides: Any) -> TuningOffsetStage:
    return TuningOffsetStage(TuningOffsetConfig(cents=cents, **overrides))


def _config_payload(effects: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "render_pipeline": {
            "synth_backend": "fluidsynth",
            "effects_chain": "pedalboard",
            "input_type": "audio",
            "sample_rate": SAMPLE_RATE,
            "bit_depth": 24,
            "channels": 2,
            "duration_padding_sec": 2.0,
            "overwrite": True,
            "resume": True,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": {
            "corpus_root": "/tmp/sonitra-tuning-offset-tests",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
        "pedalboard": {"effects": effects},
    }


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno == logging.WARNING]


# ── Config ───────────────────────────────────────────────────────────


def test_tuning_offset_config_defaults() -> None:
    """Every default mirrors ``pedalboard.time_stretch``, so the knobs need no translation."""
    cfg = TuningOffsetConfig(cents=20.0)
    assert cfg.type == "TuningOffset"
    assert cfg.enabled is True
    assert cfg.cents == 20.0
    assert cfg.high_quality is True
    assert cfg.transient_mode == "crisp"
    assert cfg.transient_detector == "compound"
    assert cfg.retain_phase_continuity is True
    assert cfg.use_long_fft_window is None
    assert cfg.use_time_domain_smoothing is False
    assert cfg.preserve_formants is True


def test_tuning_offset_requires_cents() -> None:
    """``cents`` has no meaningful default, so an unset offset must fail loudly."""
    with pytest.raises(ValidationError):
        TuningOffsetConfig(type="TuningOffset")


@pytest.mark.parametrize(
    "cents",
    [1200.01, -1200.01, float("nan"), float("inf"), float("-inf")],
)
def test_tuning_offset_rejects_out_of_range_and_non_finite(cents: float) -> None:
    """Non-finite offsets come from broken config plumbing and silently ruin a condition."""
    with pytest.raises(ValidationError):
        TuningOffsetConfig(cents=cents)


@pytest.mark.parametrize("cents", [1200.0, -1200.0])
def test_tuning_offset_accepts_range_endpoints(cents: float) -> None:
    """A whole octave up or down is legitimate for transposition experiments."""
    assert TuningOffsetConfig(cents=cents).cents == cents


def test_tuning_offset_rejects_unknown_key_and_bad_literal() -> None:
    """A misspelled knob would otherwise be dropped and the condition would not match its label."""
    with pytest.raises(ValidationError):
        TuningOffsetConfig(cents=10.0, cent_offset=3.0)
    with pytest.raises(ValidationError):
        TuningOffsetConfig(cents=10.0, transient_mode="sharp")


def test_pipeline_config_accepts_tuning_offset_in_effects() -> None:
    """The offset has to survive the discriminated union, dump and re-parse unchanged."""
    cfg = PipelineConfig.model_validate(
        _config_payload(
            [
                {"type": "TuningOffset", "cents": 12.0, "transient_mode": "smooth"},
                {"type": "Gain", "gain_db": -3.0},
            ]
        )
    )
    parsed = cfg.pedalboard.effects[0]
    assert isinstance(parsed, TuningOffsetConfig)
    assert parsed.cents == 12.0
    assert parsed.transient_mode == "smooth"

    reloaded = PipelineConfig.model_validate(cfg.model_dump(mode="json"))
    assert isinstance(reloaded.pedalboard.effects[0], TuningOffsetConfig)
    assert reloaded.pedalboard.effects[0].cents == 12.0

    for bad in ({"type": "TuningOffset"}, {"type": "NotAnEffect", "cents": 5.0}):
        with pytest.raises(ConfigError):
            PipelineConfig.model_validate(_config_payload([bad]))


def test_chain_hash_includes_tuning_offset_fields() -> None:
    """Render reuse keys off this hash, so two offsets must never collide."""
    base = TuningOffsetConfig(cents=12.0)
    assert compute_chain_hash([base]) != compute_chain_hash([TuningOffsetConfig(cents=20.0)])
    assert compute_chain_hash([base]) != compute_chain_hash(
        [TuningOffsetConfig(cents=12.0, transient_mode="smooth")]
    )
    assert compute_chain_hash([base]) == compute_chain_hash([TuningOffsetConfig(cents=12.0)])


# ── Stage behaviour ──────────────────────────────────────────────────


@pytest.mark.parametrize("cents", [-40.0, -8.0, 8.0, 40.0])
def test_offset_accuracy_guard(cents: float) -> None:
    """A condition label is only true if the applied offset matches, so pin the accuracy."""
    audio = _harmonic_tone()
    measured, spread = _measured_cents(_stage(cents)(audio, SAMPLE_RATE), SAMPLE_RATE)
    assert measured == pytest.approx(cents, abs=0.5)
    assert spread < 0.5


def test_zero_cents_is_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-tune offset must not add stretch artifacts to the unprocessed control."""

    def _boom(*_args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("time_stretch must not run for a zero-cent offset")

    monkeypatch.setattr(
        "sonitra.effects.chain_builder.pedalboard.time_stretch", _boom
    )
    audio = _harmonic_tone(duration_sec=1.0)
    out = TuningOffsetStage(TuningOffsetConfig(cents=0.0))(audio, SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


@pytest.mark.parametrize("channels", [1, 2])
def test_output_shape_dtype_and_length_preserved(channels: int) -> None:
    """Downstream quality gates and onset alignment assume an unchanged frame layout."""
    audio = _stereo_noise(samples=4410, seed=channels)
    if channels == 1:
        audio = audio[:1]
    out = _stage(12.0)(audio, SAMPLE_RATE)
    assert out.shape == audio.shape
    assert out.shape == (channels, 4410)
    assert out.dtype == np.float32


def test_float64_input_is_accepted() -> None:
    """The stretcher rejects float64 outright, so the cast has to happen first."""
    audio = np.zeros((2, 4410), dtype=np.float64)
    out = _stage(12.0)(audio, SAMPLE_RATE)
    assert out.dtype == np.float32
    assert out.shape == (2, 4410)


def test_rejects_non_2d_input() -> None:
    """The channel/sample contract is ambiguous for 1-D input, so name the offending shape."""
    audio = _harmonic_tone(duration_sec=1.0).ravel()
    with pytest.raises(ValueError) as excinfo:
        _stage(20.0)(audio, SAMPLE_RATE)
    assert str(audio.shape) in str(excinfo.value)


@pytest.mark.parametrize("samples", [1, 7, 15])
def test_very_short_input(samples: int) -> None:
    """Rubber Band loses samples or flips the channel axis here; never return garbage silently."""
    audio = _stereo_noise(samples=samples, seed=samples)
    try:
        out = _stage(20.0)(audio, SAMPLE_RATE)
    except ValueError:
        return
    assert out.shape == audio.shape
    assert out.size > 0


@pytest.mark.parametrize("delta", [-7, 7])
def test_length_guard_trims_and_pads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    delta: int,
) -> None:
    """A length drift would break the duration gate, so it is corrected and reported."""
    audio = _stereo_noise(samples=4410, seed=5)

    def fake(audio_arg: np.ndarray, sample_rate: int | None = None, **kwargs: Any) -> np.ndarray:
        drifted = audio_arg.shape[1] + delta
        return np.zeros((audio_arg.shape[0], drifted), dtype=np.float32)

    monkeypatch.setattr(
        "sonitra.effects.chain_builder.pedalboard.time_stretch", fake
    )
    with caplog.at_level(logging.WARNING):
        out = _stage(20.0)(audio, SAMPLE_RATE)
    assert out.shape == (2, 4410)
    assert len(_warnings(caplog)) == 1


def test_channel_axis_change_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transposed result would be written as garbage audio, so the fail-soft loop must see it."""
    audio = _stereo_noise(samples=4410, seed=6)

    def transposing(audio_arg: np.ndarray, sample_rate: int | None = None, **kwargs: Any) -> np.ndarray:
        return np.zeros((audio_arg.shape[1], audio_arg.shape[0]), dtype=np.float32)

    monkeypatch.setattr(
        "sonitra.effects.chain_builder.pedalboard.time_stretch", transposing
    )
    with pytest.raises(ValueError):
        _stage(20.0)(audio, SAMPLE_RATE)


def test_identical_channels_stay_identical() -> None:
    """Transcribers downmix to mono, so any channel difference would comb-filter the f0."""
    mono = _harmonic_tone(duration_sec=1.0)
    out = _stage(-20.0)(np.concatenate([mono, mono], axis=0), SAMPLE_RATE)
    np.testing.assert_array_equal(out[0], out[1])


def test_deterministic() -> None:
    """Renders resume and cache on identical bytes for the same config."""
    audio = _stereo_noise(samples=SAMPLE_RATE, seed=3)
    stage = _stage(-20.0)
    np.testing.assert_array_equal(stage(audio, SAMPLE_RATE), stage(audio, SAMPLE_RATE))


def test_knobs_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-default knobs must reach the stretcher, or the methods section would misdescribe the render."""
    cfg = TuningOffsetConfig(
        cents=25.0,
        high_quality=False,
        transient_mode="smooth",
        transient_detector="percussive",
        retain_phase_continuity=False,
        use_long_fft_window=True,
        use_time_domain_smoothing=True,
        preserve_formants=False,
    )
    stage = TuningOffsetStage(cfg)
    assert stage.config is cfg
    captured: list[dict[str, Any]] = []

    def fake(audio_arg: np.ndarray, sample_rate: int | None = None, **kwargs: Any) -> np.ndarray:
        captured.append(kwargs)
        return np.asarray(audio_arg, dtype=np.float32)

    monkeypatch.setattr(
        "sonitra.effects.chain_builder.pedalboard.time_stretch", fake
    )
    stage(_stereo_noise(samples=4410, seed=9), SAMPLE_RATE)

    assert len(captured) == 1
    sent = captured[0]
    assert sent["stretch_factor"] == 1.0
    assert sent["pitch_shift_in_semitones"] == pytest.approx(cfg.cents / 100)
    assert sent["high_quality"] is False
    assert sent["transient_mode"] == "smooth"
    assert sent["transient_detector"] == "percussive"
    assert sent["retain_phase_continuity"] is False
    assert sent["use_long_fft_window"] is True
    assert sent["use_time_domain_smoothing"] is True
    assert sent["preserve_formants"] is False


@pytest.mark.parametrize("cents", [50.0, -50.0])
def test_half_semitone_warning_at_build(
    caplog: pytest.LogCaptureFixture, cents: float
) -> None:
    """At half a semitone the neighbouring pitch becomes plausible, which the config author must know."""
    with caplog.at_level(logging.WARNING):
        build_effects_chain([TuningOffsetConfig(cents=cents)])
    assert len(_warnings(caplog)) == 1


def test_below_half_semitone_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Realistic offsets are the common case; warning on them would train readers to ignore warnings."""
    with caplog.at_level(logging.WARNING):
        build_effects_chain([TuningOffsetConfig(cents=49.9)])
    assert _warnings(caplog) == []


def test_disabled_tuning_offset_is_skipped() -> None:
    """Config files declare an offset slot up front and switch it on per condition."""
    assert len(build_effects_chain([TuningOffsetConfig(cents=20.0, enabled=False)])) == 0


# ── Composite chain ──────────────────────────────────────────────────


def test_chain_without_tuning_is_bit_identical_to_pedalboard() -> None:
    """Without an offset the chain must be one native segment, so no existing render changes."""
    x = _stereo_noise(samples=4410, seed=7)
    out = build_effects_chain([_REVERB, _DISTORTION, _COMPRESSOR])(x, SAMPLE_RATE)
    reference = pedalboard.Pedalboard(
        [
            pedalboard.Reverb(
                room_size=_REVERB.room_size,
                damping=_REVERB.damping,
                wet_level=_REVERB.wet_level,
                dry_level=_REVERB.dry_level,
                width=_REVERB.width,
                freeze_mode=_REVERB.freeze_mode,
            ),
            pedalboard.Distortion(drive_db=_DISTORTION.drive_db),
            pedalboard.Compressor(
                threshold_db=_COMPRESSOR.threshold_db,
                ratio=_COMPRESSOR.ratio,
                attack_ms=_COMPRESSOR.attack_ms,
                release_ms=_COMPRESSOR.release_ms,
            ),
        ]
    )(x, SAMPLE_RATE)
    np.testing.assert_array_equal(out, reference)


def test_chain_order_with_tuning_mid_chain() -> None:
    """Splitting the chain into segments must not reorder or merge anything."""
    x = _stereo_noise(samples=4410, seed=11)
    tuning = TuningOffsetConfig(cents=20.0)
    out = build_effects_chain([_DISTORTION, tuning, _REVERB])(x, SAMPLE_RATE)

    after_distortion = build_effects_chain([_DISTORTION])[0](x, SAMPLE_RATE)
    after_offset = TuningOffsetStage(tuning)(after_distortion, SAMPLE_RATE)
    reference = build_effects_chain([_REVERB])[0](after_offset, SAMPLE_RATE)
    np.testing.assert_array_equal(out, reference)


def test_chain_len_and_indexing() -> None:
    """The pipeline relies on ``len(chain)`` and callers inspect stages by index."""
    chain = build_effects_chain(
        [
            GainConfig(gain_db=-2.0),
            TuningOffsetConfig(cents=20.0),
            DistortionConfig(drive_db=18.0, enabled=False),
            LimiterConfig(threshold_db=-1.0, release_ms=100.0),
        ]
    )
    assert type(chain) is CompositeEffectsChain
    assert len(chain) == 3
    assert isinstance(chain[0], pedalboard.Gain)
    assert isinstance(chain[1], TuningOffsetStage)
    assert isinstance(chain[2], pedalboard.Limiter)
    indexed = [chain[0], chain[1], chain[2]]
    assert len(list(chain)) == 3
    assert all(iterated is expected for iterated, expected in zip(chain, indexed))


def test_chain_satisfies_effects_chain_protocol() -> None:
    """The protocol check is only meaningful if a plain callable does not satisfy it."""
    chain = build_effects_chain([_DISTORTION])

    def plain(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio

    assert isinstance(chain, EffectsChain)
    assert not isinstance(plain, EffectsChain)


def test_segments_called_positionally(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Pedalboard.__call__`` only binds ``(audio, sample_rate)`` positionally."""
    observed: list[tuple[int, int]] = []
    original_call = pedalboard.Pedalboard.__call__

    def record(audio: np.ndarray, sr: int) -> None:
        observed.append((int(audio.shape[1]), int(sr)))

    monkeypatch.setattr(
        pedalboard.Pedalboard,
        "__call__",
        lambda self, audio, sr: record(audio, sr) or original_call(self, audio, sr),
    )
    x = _stereo_noise(samples=4410, seed=13)
    chain = build_effects_chain(
        [
            GainConfig(gain_db=6.0),
            TuningOffsetConfig(cents=20.0),
            LimiterConfig(threshold_db=-1.0, release_ms=50.0),
        ]
    )
    # The offset is its own segment: it cannot join a Pedalboard, which only
    # accepts native plugins. So the three stages yield two Pedalboard calls.
    assert len(chain) == 3
    out = chain(x, SAMPLE_RATE)

    assert [sr for _, sr in observed] == [SAMPLE_RATE, SAMPLE_RATE]
    assert [length for length, _ in observed] == [x.shape[1]] * 2
    assert out.shape == x.shape
    assert not np.array_equal(out, x)


# ── End to end ───────────────────────────────────────────────────────


def test_pipeline_renders_with_tuning_offset(tmp_path: Path) -> None:
    """A condition is only real once the render on disk carries the declared offset."""
    tone = _harmonic_tone(duration_sec=2.0)
    stereo = np.concatenate([tone, tone], axis=0)
    # normalize=False: keep the tone's own (well below full-scale) peak so
    # the default quality-gate clip_threshold doesn't trip on test fixtures.
    wav = write_wav(
        stereo, tmp_path / "in.wav", sample_rate=SAMPLE_RATE, normalize=False
    )
    cfg = _audio_config(
        tmp_path,
        effects_chain="pedalboard",
        pedalboard_effects=[{"type": "TuningOffset", "cents": 20.0}],
    )

    out_dir = tmp_path / "out"
    result = run_pipeline([wav], out_dir, config=cfg)

    assert result.succeeded == 1
    assert result.failed == 0
    output_path = Path(result.log[0]["output"])
    assert output_path.exists()
    audio, sample_rate = read_audio(output_path)
    measured, _spread = _measured_cents(audio, sample_rate)
    assert measured == pytest.approx(20.0, abs=0.5)
