"""Builds the runtime effects chain from the config tree.

Chains are composite because a global tuning offset cannot live inside a
``pedalboard.Pedalboard``: it is applied by the offline stretcher, which is a
plain function. Chains without an offset collapse to a single ``Pedalboard``
segment, so their output is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from typing import Sequence, Union

import numpy as np
import pedalboard

from sonitra.config import PipelineConfig
from sonitra.effects.builtin_effects import (
    ChorusConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    EffectConfig,
    GainConfig,
    HighpassFilterConfig,
    HighShelfFilterConfig,
    LimiterConfig,
    LowpassFilterConfig,
    LowShelfFilterConfig,
    PeakFilterConfig,
    ReverbConfig,
    TuningOffsetConfig,
    VST3PluginConfig,
)

logger = logging.getLogger(__name__)

_Segment = Union[pedalboard.Pedalboard, "TuningOffsetStage"]


class TuningOffsetStage:
    """Shifts the whole signal by a fixed number of cents.

    The offset goes through ``pedalboard.time_stretch`` because the
    ``PitchShift`` plugin does not apply cent-level shifts reliably: shifts
    below about 8 cents come out as no shift at all.
    """

    def __init__(self, config: TuningOffsetConfig) -> None:
        self.config = config

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return ``audio`` shifted by the configured offset, same shape and dtype."""
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 2:
            raise ValueError(
                f"TuningOffset expects (channels, samples) audio, got shape {samples.shape}"
            )
        if self.config.cents == 0:
            # An enabled zero-cent slot is the unprocessed control, so it must
            # not pick up any stretcher artifact.
            return samples
        shifted = np.asarray(
            pedalboard.time_stretch(
                samples,
                sample_rate,
                stretch_factor=1.0,
                pitch_shift_in_semitones=self.config.cents / 100.0,
                high_quality=self.config.high_quality,
                transient_mode=self.config.transient_mode,
                transient_detector=self.config.transient_detector,
                retain_phase_continuity=self.config.retain_phase_continuity,
                use_long_fft_window=self.config.use_long_fft_window,
                use_time_domain_smoothing=self.config.use_time_domain_smoothing,
                preserve_formants=self.config.preserve_formants,
            ),
            dtype=np.float32,
        )
        return self._restore_length(shifted, samples)

    def _restore_length(
        self, shifted: np.ndarray, samples: np.ndarray
    ) -> np.ndarray:
        """Undo a samples-axis drift; refuse a channel-axis flip.

        Rubber Band loses samples or flips the channel axis on inputs of a
        sample or two, which would hand the quality gate a transposed buffer.
        Realistic input never trips this; it only exists to keep a degenerate
        file from being written as garbage.
        """
        channels, length = samples.shape
        if shifted.ndim != 2 or shifted.shape[0] != channels:
            raise ValueError(
                f"TuningOffset channel count changed from {channels} to "
                f"{shifted.shape}: a transposed result cannot be used"
            )
        produced = shifted.shape[1]
        if produced == length:
            return shifted
        logger.warning(
            "TuningOffset output length %d differs from input length %d; "
            "trimmed or zero-padded to the input length",
            produced,
            length,
        )
        restored = np.zeros((channels, length), dtype=np.float32)
        keep = min(produced, length)
        restored[:, :keep] = shifted[:, :keep]
        return restored


class CompositeEffectsChain:
    """Ordered effects chain; runs of native plugins share one ``Pedalboard`` segment."""

    def __init__(self, stages: Sequence[object]) -> None:
        self._stages = list(stages)
        self._segments: list[_Segment] = []
        pending: list[object] = []
        for stage in self._stages:
            if isinstance(stage, TuningOffsetStage):
                if pending:
                    self._segments.append(pedalboard.Pedalboard(pending))
                    pending = []
                self._segments.append(stage)
            else:
                pending.append(stage)
        if pending:
            self._segments.append(pedalboard.Pedalboard(pending))

    def __len__(self) -> int:
        return len(self._stages)

    def __getitem__(self, index: int):
        return self._stages[index]

    def __iter__(self) -> Iterator[object]:
        return iter(self._stages)

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        result = audio
        for segment in self._segments:
            result = segment(result, sample_rate)
        return result


def build_effects_chain(effects: Iterable[EffectConfig]) -> CompositeEffectsChain:
    stages: list[object] = []
    for effect_cfg in effects:
        if not effect_cfg.enabled:
            continue
        if (
            isinstance(effect_cfg, TuningOffsetConfig)
            and abs(effect_cfg.cents) >= 50.0
        ):
            # Not an error: whole-semitone offsets are legitimate transposition
            # experiments, but the neighbouring pitch becomes a real alternative.
            logger.warning(
                "TuningOffset %.1f cents is at or beyond half a semitone: "
                "notes may score as the neighbouring pitch",
                effect_cfg.cents,
            )
        stages.append(_build_effect(effect_cfg))
    return CompositeEffectsChain(stages)


def build_effects_chain_from_config(cfg: PipelineConfig) -> CompositeEffectsChain:
    return build_effects_chain(cfg.pedalboard.effects)


def compute_chain_hash(effects: Iterable[EffectConfig]) -> str:
    payload = [effect.model_dump(mode="json") for effect in effects]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_effect(effect_cfg: EffectConfig):
    if isinstance(effect_cfg, CompressorConfig):
        plugin = pedalboard.Compressor()
        plugin.threshold_db = effect_cfg.threshold_db
        plugin.ratio = effect_cfg.ratio
        plugin.attack_ms = effect_cfg.attack_ms
        plugin.release_ms = effect_cfg.release_ms
        return plugin
    if isinstance(effect_cfg, ReverbConfig):
        plugin = pedalboard.Reverb()
        plugin.room_size = effect_cfg.room_size
        plugin.damping = effect_cfg.damping
        plugin.wet_level = effect_cfg.wet_level
        plugin.dry_level = effect_cfg.dry_level
        plugin.width = effect_cfg.width
        plugin.freeze_mode = effect_cfg.freeze_mode
        return plugin
    if isinstance(effect_cfg, LimiterConfig):
        plugin = pedalboard.Limiter()
        plugin.threshold_db = effect_cfg.threshold_db
        plugin.release_ms = effect_cfg.release_ms
        return plugin
    if isinstance(effect_cfg, ChorusConfig):
        plugin = pedalboard.Chorus()
        plugin.rate_hz = effect_cfg.rate_hz
        plugin.depth = effect_cfg.depth
        plugin.centre_delay_ms = effect_cfg.centre_delay_ms
        plugin.feedback = effect_cfg.feedback
        plugin.mix = effect_cfg.mix
        return plugin
    if isinstance(effect_cfg, DelayConfig):
        plugin = pedalboard.Delay()
        plugin.delay_seconds = effect_cfg.delay_seconds
        plugin.feedback = effect_cfg.feedback
        plugin.mix = effect_cfg.mix
        return plugin
    if isinstance(effect_cfg, DistortionConfig):
        plugin = pedalboard.Distortion()
        plugin.drive_db = effect_cfg.drive_db
        return plugin
    if isinstance(effect_cfg, GainConfig):
        plugin = pedalboard.Gain()
        plugin.gain_db = effect_cfg.gain_db
        return plugin
    if isinstance(effect_cfg, VST3PluginConfig):
        plugin = pedalboard.load_plugin(str(effect_cfg.plugin_path))
        if not getattr(plugin, "is_effect", False):
            raise ValueError("VST3 plugin is not an effect")
        return plugin
    if isinstance(effect_cfg, HighpassFilterConfig):
        plugin = pedalboard.HighpassFilter()
        plugin.cutoff_frequency_hz = effect_cfg.cutoff_frequency_hz
        return plugin
    if isinstance(effect_cfg, LowpassFilterConfig):
        plugin = pedalboard.LowpassFilter()
        plugin.cutoff_frequency_hz = effect_cfg.cutoff_frequency_hz
        return plugin
    if isinstance(effect_cfg, HighShelfFilterConfig):
        plugin = pedalboard.HighShelfFilter()
        plugin.cutoff_frequency_hz = effect_cfg.cutoff_frequency_hz
        plugin.gain_db = effect_cfg.gain_db
        plugin.q = effect_cfg.q
        return plugin
    if isinstance(effect_cfg, LowShelfFilterConfig):
        plugin = pedalboard.LowShelfFilter()
        plugin.cutoff_frequency_hz = effect_cfg.cutoff_frequency_hz
        plugin.gain_db = effect_cfg.gain_db
        plugin.q = effect_cfg.q
        return plugin
    if isinstance(effect_cfg, PeakFilterConfig):
        plugin = pedalboard.PeakFilter()
        plugin.cutoff_frequency_hz = effect_cfg.cutoff_frequency_hz
        plugin.gain_db = effect_cfg.gain_db
        plugin.q = effect_cfg.q
        return plugin
    if isinstance(effect_cfg, TuningOffsetConfig):
        return TuningOffsetStage(effect_cfg)
    raise ValueError(f"Unsupported effect type: {type(effect_cfg)}")
