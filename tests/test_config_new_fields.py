from __future__ import annotations

import os

import pytest
from rich.console import Console

from sonitra.benchmark.results import WorkerEvent
from sonitra.config import (
    ConfigError,
    EffectsChain,
    FluidSynthSection,
    ObservabilitySection,
    PipelineConfig,
    SynthBackend,
    load_config,
)
from sonitra.terminal import (
    NullBenchmarkProgress,
    RichBenchmarkProgress,
    effective_log_level,
    get_console,
)


def _base_dict(synth_backend: str, effects_chain: str, **overrides) -> dict:
    base = {
        "pipeline": {
            "synth_backend": synth_backend,
            "effects_chain": effects_chain,
            "bpm": 120,
            "sample_rate": 44100,
            "bit_depth": 24,
            "channels": 2,
            "duration_padding_sec": 2.0,
            "overwrite": False,
            "resume": True,
            "max_workers": 1,
            "log_level": "INFO",
        },
        "io": {
            "corpus_root": ".",
            "output_format": "wav",
            "mp3_bitrate_kbps": 192,
            "file_naming": "{stem}",
        },
    }
    base.update(overrides)
    return base


def test_synth_backend_enum_values() -> None:
    assert SynthBackend.FLUIDSYNTH.value == "fluidsynth"
    assert SynthBackend.DAWDREAMER_FAUST.value == "dawdreamer_faust"
    assert SynthBackend.DAWDREAMER_VST.value == "dawdreamer_vst"
    assert SynthBackend.PEDALBOARD_INSTRUMENT.value == "pedalboard_instrument"

def test_effects_chain_enum_values() -> None:
    assert EffectsChain.NONE.value == "none"
    assert EffectsChain.PEDALBOARD.value == "pedalboard"

def test_all_synth_backends_parse() -> None:
    for backend in SynthBackend:
        extra: dict = {}
        if backend == SynthBackend.FLUIDSYNTH:
            extra = {"fluidsynth": {"soundfont_path": "/tmp/dummy.sf2"}}
        elif backend == SynthBackend.DAWDREAMER_VST:
            extra = {"dawdreamer": {"plugin_path": "/tmp/dummy.vst3"}}
        cfg = PipelineConfig.model_validate(
            _base_dict(backend.value, "none", **extra)
        )
        assert cfg.pipeline.synth_backend == backend

def test_all_effects_chains_parse() -> None:
    for chain in EffectsChain:
        cfg = PipelineConfig.model_validate(
            _base_dict("dawdreamer_faust", chain.value)
        )
        assert cfg.pipeline.effects_chain == chain

def test_invalid_synth_backend_raises() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(_base_dict("invalid_mode", "none"))

def test_fluidsynth_without_soundfont_raises() -> None:
    data = _base_dict("fluidsynth", "none", fluidsynth={"soundfont_path": None})
    with pytest.raises(ConfigError, match="soundfont_path"):
        PipelineConfig.model_validate(data)

def test_fluidsynth_with_soundfont_passes() -> None:
    data = _base_dict("fluidsynth", "none",
                      fluidsynth={"soundfont_path": "/tmp/dummy.sf2"})
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.synth_backend == SynthBackend.FLUIDSYNTH

def test_fluidsynth_without_soundfont_section_raises() -> None:
    data = _base_dict("fluidsynth", "none")
    with pytest.raises(ConfigError, match="soundfont_path"):
        PipelineConfig.model_validate(data)

def test_dawdreamer_vst_without_plugin_path_raises() -> None:
    data = _base_dict("dawdreamer_vst", "none")
    with pytest.raises(ConfigError, match="plugin_path"):
        PipelineConfig.model_validate(data)

def test_dawdreamer_vst_with_plugin_path_passes() -> None:
    data = _base_dict("dawdreamer_vst", "none",
                      dawdreamer={"plugin_path": "/tmp/dummy.vst3"})
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.synth_backend == SynthBackend.DAWDREAMER_VST

def test_dawdreamer_faust_with_plugin_path_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none",
                      dawdreamer={"plugin_path": "/tmp/dummy.vst3"})
    with pytest.raises(ConfigError, match="plugin_path"):
        PipelineConfig.model_validate(data)

def test_dawdreamer_faust_without_plugin_path_passes() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("dawdreamer_faust", "none"))
    assert cfg.pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST

def test_pedalboard_instrument_without_plugin_path_passes() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("pedalboard_instrument", "none"))
    assert cfg.pipeline.synth_backend == SynthBackend.PEDALBOARD_INSTRUMENT

def test_rendering_mode_in_pipeline_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["rendering_mode"] = "dawdreamer_only"
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_dawdreamer_enabled_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["dawdreamer"] = {"enabled": True, "block_size": 512}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_pedalboard_enabled_raises() -> None:
    data = _base_dict("pedalboard_instrument", "pedalboard")
    data["pedalboard"] = {"enabled": True}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_dawdreamer_soundfont_path_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["dawdreamer"] = {"soundfont_path": "/tmp/a.sf2"}
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_fluidsynthsection_extra_key_raises() -> None:
    data = _base_dict("fluidsynth", "none",
                      fluidsynth={"soundfont_path": "/tmp/dummy.sf2", "bogus": True})
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_validate_worker_constraint_forces_1_for_dawdreamer_faust() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("dawdreamer_faust", "none"))
    cfg.pipeline.max_workers = 8
    result = cfg.validate_worker_constraint()
    assert result.pipeline.max_workers == 1

def test_validate_worker_constraint_forces_1_for_dawdreamer_vst() -> None:
    data = _base_dict("dawdreamer_vst", "none",
                      dawdreamer={"plugin_path": "/tmp/x.vst3"})
    cfg = PipelineConfig.model_validate(data)
    cfg.pipeline.max_workers = 4
    assert cfg.validate_worker_constraint().pipeline.max_workers == 1

def test_validate_worker_constraint_allows_multiple_for_pedalboard_instrument() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("pedalboard_instrument", "pedalboard"))
    cfg.pipeline.max_workers = 4
    assert cfg.validate_worker_constraint().pipeline.max_workers == 4

def test_validate_worker_constraint_allows_multiple_for_fluidsynth() -> None:
    data = _base_dict("fluidsynth", "none",
                      fluidsynth={"soundfont_path": "/tmp/x.sf2"})
    cfg = PipelineConfig.model_validate(data)
    cfg.pipeline.max_workers = 4
    assert cfg.validate_worker_constraint().pipeline.max_workers == 4

def test_bpm_defaults_to_120() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"].pop("bpm", None)
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.bpm == 120

def test_bpm_zero_rejected() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["bpm"] = 0
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_bpm_negative_rejected() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["bpm"] = -10
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)

def test_bpm_1_is_valid() -> None:
    data = _base_dict("dawdreamer_faust", "none")
    data["pipeline"]["bpm"] = 1
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pipeline.bpm == 1


# ── Observability: log_level + progress ──────────────────────────────

def test_observability_progress_false_accepted() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict("dawdreamer_faust", "none", observability={"progress": False})
    )
    assert cfg.observability.progress is False


def test_observability_log_level_debug_accepted() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict("dawdreamer_faust", "none", observability={"log_level": "DEBUG"})
    )
    assert cfg.observability.log_level == "DEBUG"


def test_observability_log_level_lowercase_normalized() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict("dawdreamer_faust", "none", observability={"log_level": "warning"})
    )
    assert cfg.observability.log_level == "WARNING"


def test_observability_log_level_null_accepted() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict("dawdreamer_faust", "none", observability={"log_level": None})
    )
    assert cfg.observability.log_level is None


def test_observability_log_level_invalid_raises() -> None:
    data = _base_dict("dawdreamer_faust", "none", observability={"log_level": "SILLY"})
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(data)


def test_observability_section_direct_construction() -> None:
    section = ObservabilitySection(log_level="debug", progress=False)
    assert section.log_level == "DEBUG"
    assert section.progress is False


def test_observability_defaults() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("dawdreamer_faust", "none"))
    assert cfg.observability.log_level is None
    assert cfg.observability.progress is True


# ── effective_log_level fallback ─────────────────────────────────────

def test_effective_log_level_prefers_observability() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict("dawdreamer_faust", "none", observability={"log_level": "debug"})
    )
    assert effective_log_level(cfg) == "DEBUG"


def test_effective_log_level_falls_back_to_pipeline() -> None:
    cfg = PipelineConfig.model_validate(
        _base_dict("dawdreamer_faust", "none", observability={"log_level": None})
    )
    cfg.pipeline.log_level = "warning"
    assert effective_log_level(cfg) == "WARNING"


def test_effective_log_level_defaults_to_info() -> None:
    cfg = PipelineConfig.model_validate(_base_dict("dawdreamer_faust", "none"))
    cfg.observability.log_level = None
    cfg.pipeline.log_level = ""
    assert effective_log_level(cfg) == "INFO"


# ── Benchmark progress displays ──────────────────────────────────────

def _worker_event(**overrides) -> WorkerEvent:
    fields = {
        "worker_id": os.getpid(),
        "condition": "baseline",
        "transcriber": "basic_pitch",
        "midi_path": "song.mid",
        "status": "start",
        "ok": True,
    }
    fields.update(overrides)
    return WorkerEvent(**fields)


def test_null_benchmark_progress_methods_are_noops() -> None:
    progress = NullBenchmarkProgress()
    progress.on_condition_started("baseline", {"a": 1}, 3, ["basic_pitch"])
    progress.on_worker_event(_worker_event(status="start"))
    progress.on_worker_event(_worker_event(status="done", ok=False))
    progress.on_condition_done("baseline")


def test_null_benchmark_progress_worker_event_is_noop() -> None:
    progress = NullBenchmarkProgress()
    assert progress.on_worker_event(_worker_event(status="done", ok=False)) is None


def test_rich_benchmark_progress_enters_exits_with_two_workers() -> None:
    progress = RichBenchmarkProgress(Console(force_terminal=True), n_workers=2)
    with progress:
        # Rows are pre-created before any events arrive.
        assert len(progress._worker_rows) == 2
        assert len(progress._workers_progress.tasks) == 2
        # The idle rows render without raising (file field present).
        progress._live.refresh()
    assert len(progress._worker_rows) == 2


def test_rich_benchmark_progress_enters_exits_and_advances() -> None:
    progress = RichBenchmarkProgress(Console(force_terminal=True), n_workers=2)
    with progress:
        progress.on_condition_started(
            "baseline",
            {"pedalboard.effects.1.wet_level": 0.5},
            2,
            ["basic_pitch"],
        )
        progress.on_worker_event(_worker_event(status="start"))
        progress.on_worker_event(_worker_event(status="done", ok=True))
        progress.on_worker_event(
            _worker_event(midi_path="song2.mid", status="start")
        )
        progress.on_worker_event(
            _worker_event(midi_path="song2.mid", status="done", ok=False)
        )
    assert progress._failed == 1
    assert progress.header.plain == (
        f"benchmark · 2/2 files · 2 workers · pids [{os.getpid()}] 1 failed"
    )


def test_rich_benchmark_progress_event_sequence_updates_display() -> None:
    """A start/done sequence must drive rows and the sweep bar without raising."""
    progress = RichBenchmarkProgress(Console(force_terminal=True), n_workers=2)
    with progress:
        progress.on_condition_started(
            "baseline", {}, 2, ["basic_pitch", "second"]
        )
        progress.on_worker_event(_worker_event(status="start"))
        progress.on_worker_event(_worker_event(status="done", ok=True))
        progress.on_worker_event(
            _worker_event(
                worker_id=os.getpid() + 1,
                transcriber="second",
                midi_path="song2.mid",
                status="start",
            )
        )
        progress.on_worker_event(
            _worker_event(
                worker_id=os.getpid() + 1,
                transcriber="second",
                midi_path="song2.mid",
                status="done",
                ok=True,
            )
        )
        # Distinct worker pids map to distinct pre-created rows.
        assert len(progress._worker_task_ids) == 2
        assert (
            progress._worker_task_ids[os.getpid()]
            != progress._worker_task_ids[os.getpid() + 1]
        )
        # A worker keeps its row for subsequent events.
        row = progress._worker_task_ids[os.getpid()]
        progress.on_worker_event(_worker_event(status="start"))
        assert progress._worker_task_ids[os.getpid()] == row
    assert progress._failed == 0
    assert progress.header.plain == (
        f"benchmark · 2/4 files · 2 workers · pids "
        f"[{os.getpid()}, {os.getpid() + 1}]"
    )


def test_rich_benchmark_progress_non_tty_smoke() -> None:
    """The display must not raise in a non-TTY environment (e.g. CI)."""
    progress = RichBenchmarkProgress(get_console(), n_workers=2)
    with progress:
        progress.on_condition_started("baseline", {}, 2, ["basic_pitch"])
        progress.on_worker_event(_worker_event(status="start"))
        progress.on_worker_event(_worker_event(status="done", ok=True))
        progress.on_worker_event(_worker_event(status="done", ok=False))
    assert progress._failed == 1


def test_rich_benchmark_progress_no_double_render() -> None:
    """Regression: only the outer Live may be entered.

    Entering the Progress objects as context managers would nest their Lives
    on ``console._live_stack`` and the outermost Live renders the whole stack,
    so every frame would render twice.
    """
    console = Console(force_terminal=True)
    progress = RichBenchmarkProgress(console, n_workers=2)
    with progress:
        # Root cause: exactly one Live on the stack, the outer one.
        assert len(console._live_stack) == 1
        assert console._live_stack[0] is progress._live
        # The Live's Group renderable holds exactly the header plus one outer
        # and one workers Progress — not two copies of each.
        group = progress._live._renderable
        renderables = list(group.renderables)
        assert len(renderables) == 3
        assert renderables[0] is progress.header
        assert renderables[1] is progress._outer_progress
        assert renderables[2] is progress._workers_progress
        # The rendered frame wraps that single Group; a buggy build would
        # append the nested Lives' renderables here (3 elements).
        rendered = progress._live.renderable
        assert len(rendered.renderables) == 1
        assert rendered.renderables[0] is group


def test_rich_benchmark_progress_lazy_start_without_context() -> None:
    progress = RichBenchmarkProgress(get_console(), n_workers=1)
    progress.on_condition_started("baseline", {}, 1, ["basic_pitch"])
    progress.on_worker_event(_worker_event(status="start"))
    progress.on_worker_event(_worker_event(status="done", ok=True))
    assert progress.header.plain == (
        f"benchmark · 1/1 files · 1 workers · pids [{os.getpid()}]"
    )
    progress.on_condition_done("baseline")
    progress.__exit__(None, None, None)


def test_rich_benchmark_progress_header_devices_and_pids() -> None:
    """Header shows the configured devices chip and the live worker pid list."""
    progress = RichBenchmarkProgress(
        Console(no_color=True), n_workers=2, devices={"basic_pitch": "cpu"}
    )
    with progress:
        progress.on_condition_started("baseline", {}, 2, ["basic_pitch"])
        # Devices known from config; no pids observed yet.
        assert (
            progress.header.plain
            == "benchmark · 0/2 files · 2 workers · devices: cpu"
        )
        progress.on_worker_event(_worker_event(worker_id=40925, status="start"))
        assert progress.header.plain == (
            "benchmark · 0/2 files · 2 workers · devices: cpu · pids [40925]"
        )
        progress.on_worker_event(_worker_event(worker_id=40926, status="start"))
        progress.on_worker_event(
            _worker_event(worker_id=40925, status="done", ok=True)
        )
        progress.on_worker_event(
            _worker_event(worker_id=40926, status="done", ok=True)
        )
    assert progress.header.plain == (
        "benchmark · 2/2 files · 2 workers · devices: cpu · pids [40925, 40926]"
    )


def test_rich_benchmark_progress_row_pid_and_device_chip() -> None:
    """Row descriptions carry the pid and a device chip when known."""
    progress = RichBenchmarkProgress(
        Console(no_color=True),
        n_workers=3,
        devices={"basic_pitch": "cpu", "second": "GPU:0"},
    )
    with progress:
        progress.on_worker_event(
            _worker_event(worker_id=1001, status="start")
        )
        progress.on_worker_event(
            _worker_event(
                worker_id=1002,
                transcriber="second",
                midi_path="song2.mid",
                status="start",
            )
        )
        # 'oracle' is not in the devices map -> no device chip.
        progress.on_worker_event(
            _worker_event(
                worker_id=1003,
                transcriber="oracle",
                midi_path="song3.mid",
                status="start",
            )
        )
        tasks = progress._workers_progress.tasks
        assert (
            tasks[progress._worker_task_ids[1001]].description
            == "pid 1001 · baseline × basic_pitch · cpu"
        )
        assert (
            tasks[progress._worker_task_ids[1002]].description
            == "pid 1002 · baseline × second · GPU:0"
        )
        assert (
            tasks[progress._worker_task_ids[1003]].description
            == "pid 1003 · baseline × oracle"
        )
    # Multi-device chips join sorted (case-insensitive) with a comma.
    assert "· devices: cpu, GPU:0" in progress.header.plain
    assert "pids [1001, 1002, 1003]" in progress.header.plain
