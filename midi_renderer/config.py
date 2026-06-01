from __future__ import annotations

from enum import Enum
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from midi_renderer.effects.builtin_effects import EffectConfig

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    pass


class RenderingMode(str, Enum):
    DAWDREAMER_ONLY = "dawdreamer_only"
    PEDALBOARD_ONLY = "pedalboard_only"
    DAWDREAMER_SYNTH_PEDALBOARD_FX = "dawdreamer_synth_pedalboard_fx"


class PipelineSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rendering_mode: RenderingMode
    sample_rate: int
    bit_depth: int
    channels: int
    duration_padding_sec: float
    overwrite: bool
    resume: bool
    max_workers: int
    log_level: str


class IOSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    midi_dir: Path | str
    output_dir: Path | str
    output_format: str
    mp3_bitrate_kbps: int
    file_naming: str

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, value: str) -> str:
        allowed = {"wav", "flac", "mp3"}
        if value not in allowed:
            raise ValueError(f"Unsupported output_format: {value}")
        return value


class DawDreamerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    block_size: int = 512
    plugin_path: Path | None = None
    preset_path: Path | None = None
    bpm: int = 120
    faust_code: str | None = None
    clear_midi_between_renders: bool = True


class PedalboardInstrumentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_path: Path | None = None
    preset_path: Path | None = None
    reload_plugin_per_file: bool = False
    silence_flush_sec: float = 0.0


class PedalboardSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    instrument: PedalboardInstrumentSection = Field(default_factory=PedalboardInstrumentSection)
    effects: list[EffectConfig] = Field(default_factory=list)


class NormalisationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: str = "peak"
    target_db: float = -1.0
    pre_effects: bool = False


class QualityGatesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    silence_threshold_rms: float = 0.0
    min_duration_sec: float = 0.0
    max_duration_deviation_sec: float = 0.0
    clip_threshold: float = 1.0


class ObservabilitySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write_manifest: bool = False
    manifest_path: Path | str = "renders.jsonl"
    write_failed_list: bool = False
    emit_sse_events: bool = False


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineSection
    io: IOSection
    dawdreamer: DawDreamerSection = Field(default_factory=DawDreamerSection)
    pedalboard: PedalboardSection = Field(default_factory=PedalboardSection)
    normalisation: NormalisationSection = Field(default_factory=NormalisationSection)
    quality_gates: QualityGatesSection = Field(default_factory=QualityGatesSection)
    observability: ObservabilitySection = Field(default_factory=ObservabilitySection)

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "PipelineConfig":
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

    def validate_worker_constraint(self) -> "PipelineConfig":
        if self.pipeline.rendering_mode in {
            RenderingMode.DAWDREAMER_ONLY,
            RenderingMode.DAWDREAMER_SYNTH_PEDALBOARD_FX,
        } and self.pipeline.max_workers != 1:
            logger.warning("max_workers forced to 1 for dawdreamer modes")
            self.pipeline.max_workers = 1
        return self

    def save(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        output_path.write_text(yaml.safe_dump(data, sort_keys=False))


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path | str) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError("Invalid YAML") from exc
    if payload is None:
        raise ConfigError("Empty configuration")
    try:
        return PipelineConfig.model_validate(payload)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(str(exc)) from exc