from __future__ import annotations

import hashlib
import json
from typing import Iterable

import pedalboard

from sonitra.config import PipelineConfig
from sonitra.effects.builtin_effects import (
    ChorusConfig,
    CompressorConfig,
    DelayConfig,
    DistortionConfig,
    EffectConfig,
    GainConfig,
    LimiterConfig,
    ReverbConfig,
    VST3PluginConfig,
)


def build_effects_chain(effects: Iterable[EffectConfig]) -> pedalboard.Pedalboard:
    board = pedalboard.Pedalboard()
    for effect_cfg in effects:
        if not effect_cfg.enabled:
            continue
        board.append(_build_effect(effect_cfg))
    return board


def build_effects_chain_from_config(cfg: PipelineConfig) -> pedalboard.Pedalboard:
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
    raise ValueError(f"Unsupported effect type: {type(effect_cfg)}")
