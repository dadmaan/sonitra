from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sonitra.effects.builtin_effects import EffectConfig
from sonitra.transcribe.configs import TranscriberConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusPaths:
    midi: Path
    audio: Path
    transcription: Path
    eval_results: Path


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

    corpus_root: Path | str = "corpus"
    output_format: str
    mp3_bitrate_kbps: int
    file_naming: str
    dataset: str | None = None

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
    soundfont_path: Path | None = None
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


class SeparationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: str = "passthrough"
    model: str = "htdemucs"
    device: str = "cpu"
    stem: str | None = None
    output_dir: Path | str = "stems"


class TranscriptionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcribers: list[TranscriberConfig] = Field(default_factory=list)
    output_dir: Path | str = "transcriptions"


class NoteMetricsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    onset_tolerance_sec: float = 0.05
    offset_ratio: float = 0.2
    offset_min_tolerance_sec: float = 0.05
    velocity_tolerance: float = 0.1


class FrameMetricsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    hop_sec: float = 0.01


class ExpressiveMetricsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    harmony_window_sec: float = 2.0


class DTWMetricSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    frame_size: int = 4096
    hop_size: int = 2048
    max_frames: int = 4000


class EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_metrics: NoteMetricsSection = Field(default_factory=NoteMetricsSection)
    frame_metrics: FrameMetricsSection = Field(default_factory=FrameMetricsSection)
    expressive_metrics: ExpressiveMetricsSection = Field(default_factory=ExpressiveMetricsSection)
    dtw: DTWMetricSection = Field(default_factory=DTWMetricSection)


class ConditionSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class SweepSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    values: list[Any]
    name: str | None = None


class BenchmarkSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_path: Path | str = "benchmark_results.jsonl"
    include_baseline: bool = True
    baseline_name: str = "baseline"
    conditions: list[ConditionSection] = Field(default_factory=list)
    sweeps: list[SweepSection] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineSection
    io: IOSection
    dawdreamer: DawDreamerSection = Field(default_factory=DawDreamerSection)
    pedalboard: PedalboardSection = Field(default_factory=PedalboardSection)
    normalisation: NormalisationSection = Field(default_factory=NormalisationSection)
    quality_gates: QualityGatesSection = Field(default_factory=QualityGatesSection)
    observability: ObservabilitySection = Field(default_factory=ObservabilitySection)
    separation: SeparationSection = Field(default_factory=SeparationSection)
    transcription: TranscriptionSection = Field(default_factory=TranscriptionSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    benchmark: BenchmarkSection = Field(default_factory=BenchmarkSection)

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
    return Path(__file__).resolve().parents[2] / "config" / "source.yaml"


def resolve_corpus_paths(
    cfg: PipelineConfig,
    config_name: str | None = None,
) -> CorpusPaths:
    """Return corpus subdirectory paths derived from ``cfg.io.corpus_root``.

    Args:
        cfg: Loaded pipeline configuration.
        config_name: Stem of the YAML config filename (e.g. ``"pedalboard_baseline"``
            for ``config/pedalboard_baseline.yaml``).  When provided, the audio and
            transcription directories gain a per-config suffix so different render
            configurations are kept separate.

    Returns:
        A :class:`CorpusPaths` instance with ``midi``, ``audio``, ``transcription``,
        and ``eval_results`` attributes as :class:`pathlib.Path` objects.

        With ``dataset="test"``, ``corpus_root="./corpus"``, and
        ``config_name="pedalboard_baseline"``:

        - ``midi``          → ``corpus/test/midi``
        - ``audio``         → ``corpus/test/audio/pedalboard_baseline``
        - ``transcription`` → ``corpus/test/transcription/pedalboard_baseline``
        - ``eval_results``  → ``corpus/test/eval_results``

        Without ``dataset``:

        - ``midi``          → ``corpus/midi``
        - ``audio``         → ``corpus/audio/pedalboard_baseline``
        - ``transcription`` → ``corpus/transcription/pedalboard_baseline``
        - ``eval_results``  → ``corpus/eval_results``
    """
    root = Path(cfg.io.corpus_root)
    base = root / cfg.io.dataset if cfg.io.dataset else root
    return CorpusPaths(
        midi=base / "midi",
        audio=(base / "audio" / config_name) if config_name else base / "audio",
        transcription=(base / "transcription" / config_name) if config_name else base / "transcription",
        eval_results=base / "eval_results",
    )


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