from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _EffectBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class CompressorConfig(_EffectBase):
    type: Literal["Compressor"] = "Compressor"
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float


class ReverbConfig(_EffectBase):
    type: Literal["Reverb"] = "Reverb"
    room_size: float
    damping: float
    wet_level: float
    dry_level: float
    width: float
    freeze_mode: bool


class LimiterConfig(_EffectBase):
    type: Literal["Limiter"] = "Limiter"
    threshold_db: float
    release_ms: float


class ChorusConfig(_EffectBase):
    type: Literal["Chorus"] = "Chorus"
    rate_hz: float
    depth: float
    centre_delay_ms: float
    feedback: float
    mix: float


class DelayConfig(_EffectBase):
    type: Literal["Delay"] = "Delay"
    delay_seconds: float
    feedback: float
    mix: float


class DistortionConfig(_EffectBase):
    type: Literal["Distortion"] = "Distortion"
    drive_db: float


class GainConfig(_EffectBase):
    type: Literal["Gain"] = "Gain"
    gain_db: float


class VST3PluginConfig(_EffectBase):
    type: Literal["VST3Plugin"] = "VST3Plugin"
    plugin_path: Path | str


class HighpassFilterConfig(_EffectBase):
    type: Literal["HighpassFilter"] = "HighpassFilter"
    cutoff_frequency_hz: float


class LowpassFilterConfig(_EffectBase):
    type: Literal["LowpassFilter"] = "LowpassFilter"
    cutoff_frequency_hz: float


class HighShelfFilterConfig(_EffectBase):
    type: Literal["HighShelfFilter"] = "HighShelfFilter"
    cutoff_frequency_hz: float
    gain_db: float
    q: float


class LowShelfFilterConfig(_EffectBase):
    type: Literal["LowShelfFilter"] = "LowShelfFilter"
    cutoff_frequency_hz: float
    gain_db: float
    q: float


class PeakFilterConfig(_EffectBase):
    type: Literal["PeakFilter"] = "PeakFilter"
    cutoff_frequency_hz: float
    gain_db: float
    q: float


EffectConfig = Annotated[
    Union[
        CompressorConfig,
        ReverbConfig,
        LimiterConfig,
        ChorusConfig,
        DelayConfig,
        DistortionConfig,
        GainConfig,
        VST3PluginConfig,
        HighpassFilterConfig,
        LowpassFilterConfig,
        HighShelfFilterConfig,
        LowShelfFilterConfig,
        PeakFilterConfig,
    ],
    Field(discriminator="type"),
]
