from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any
import time

import numpy as np
import pedalboard

from midi_renderer.config import PipelineConfig, RenderingMode
from midi_renderer.engine import RendererEngine
from midi_renderer.effects.chain_builder import build_effects_chain_from_config, compute_chain_hash
from midi_renderer.manifest import ManifestEntry, ManifestWriter
from midi_renderer.midi_reader import parse_midi
from midi_renderer.normaliser import normalise_from_config
from midi_renderer.quality_gate import check_quality
from midi_renderer.renderer import render_notes_faust, render_notes_vst
from midi_renderer.storage import derive_output_path, write_audio, write_wav
from midi_renderer.synth.protocol import make_synth


@dataclass
class PipelineResult:
    succeeded: int
    failed: int
    skipped: int
    elapsed_seconds: float
    log: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_seconds": self.elapsed_seconds,
            "log": self.log,
        }


def run_pipeline(
    midi_paths: Iterable[Path | str],
    out_dir: Path | str,
    engine: RendererEngine | None = None,
    plugin_path: Path | str | None = None,
    *,
    overwrite: bool = True,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0
    log: List[Dict[str, Any]] = []

    if config is not None:
        cfg = config.validate_worker_constraint()
        _get_worker_count(cfg)
        synth = make_synth(cfg)
        chain = build_effects_chain_from_config(cfg)
        chain_hash = compute_chain_hash(cfg.pedalboard.effects)
        manifest = (
            ManifestWriter(
                cfg.observability.manifest_path,
                failed_list_path=str(Path(cfg.observability.manifest_path).with_suffix(".failed.txt"))
                if cfg.observability.write_failed_list
                else None,
            )
            if cfg.observability.write_manifest
            else None
        )

        for midi_path in midi_paths:
            midi_path = Path(midi_path)
            output_path = _resolve_output_path(midi_path, out_dir, cfg)
            if output_path.exists() and not cfg.pipeline.overwrite:
                skipped += 1
                log.append({"midi": str(midi_path), "output": str(output_path), "status": "skipped"})
                continue
            try:
                notes = parse_midi(midi_path)
                duration = _compute_duration(notes, cfg.pipeline.duration_padding_sec)
                audio = synth.render(notes, duration_sec=duration)
                audio = normalise_from_config(audio, cfg, stage="pre")
                if cfg.pipeline.rendering_mode != RenderingMode.DAWDREAMER_ONLY and len(chain) > 0:
                    audio = chain(audio, cfg.pipeline.sample_rate)
                audio = normalise_from_config(audio, cfg, stage="post")

                quality = check_quality(audio, cfg.pipeline.sample_rate, cfg.quality_gates)
                if not quality.passed:
                    failed += 1
                    entry = ManifestEntry(
                        midi_path=str(midi_path),
                        output_path=str(output_path),
                        rendering_mode=cfg.pipeline.rendering_mode.value,
                        effects_chain_hash=chain_hash,
                        status="failed",
                        duration_sec=quality.duration_sec,
                        rms=quality.rms,
                        peak=quality.peak,
                        elapsed_seconds=0.0,
                        quality_flags=quality.to_dict(),
                    )
                    if manifest:
                        manifest.write(entry)
                    log.append(
                        {
                            "midi": str(midi_path),
                            "output": str(output_path),
                            "status": "failed",
                            "quality_flags": quality.to_dict(),
                        }
                    )
                    continue

                write_audio(
                    audio,
                    output_path,
                    sample_rate=cfg.pipeline.sample_rate,
                    bit_depth=cfg.pipeline.bit_depth,
                    output_format=cfg.io.output_format,
                    mp3_bitrate_kbps=cfg.io.mp3_bitrate_kbps,
                    overwrite=cfg.pipeline.overwrite,
                )
                succeeded += 1
                entry = ManifestEntry(
                    midi_path=str(midi_path),
                    output_path=str(output_path),
                    rendering_mode=cfg.pipeline.rendering_mode.value,
                    effects_chain_hash=chain_hash,
                    status="done",
                    duration_sec=quality.duration_sec,
                    rms=quality.rms,
                    peak=quality.peak,
                    elapsed_seconds=0.0,
                    quality_flags=quality.to_dict(),
                )
                if manifest:
                    manifest.write(entry)
                log.append(
                    {
                        "midi": str(midi_path),
                        "output": str(output_path),
                        "status": "succeeded",
                        "quality_flags": quality.to_dict(),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - pipeline logs and continues
                failed += 1
                if manifest:
                    manifest.write(
                        ManifestEntry(
                            midi_path=str(midi_path),
                            output_path=str(output_path),
                            rendering_mode=cfg.pipeline.rendering_mode.value,
                            effects_chain_hash=chain_hash,
                            status="failed",
                            duration_sec=0.0,
                            rms=0.0,
                            peak=0.0,
                            elapsed_seconds=0.0,
                            quality_flags={"error": str(exc)},
                        )
                    )
                log.append(
                    {
                        "midi": str(midi_path),
                        "output": str(output_path),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
        elapsed = time.perf_counter() - start
        return PipelineResult(
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            elapsed_seconds=elapsed,
            log=log,
        )

    if engine is None:
        raise ValueError("engine is required when config is not provided")

    for midi_path in midi_paths:
        midi_path = Path(midi_path)
        output_path = derive_output_path(midi_path, out_dir=out_dir, ext=".wav")
        if output_path.exists() and not overwrite:
            skipped += 1
            log.append({"midi": str(midi_path), "output": str(output_path), "status": "skipped"})
            continue

        try:
            notes = parse_midi(midi_path)
            if plugin_path:
                audio = render_notes_vst(notes, engine=engine, plugin_path=plugin_path, duration_sec=None)
            else:
                audio = render_notes_faust(notes, engine=engine, duration_sec=None)
            write_wav(audio, output_path, sample_rate=engine.sample_rate, overwrite=overwrite)
            succeeded += 1
            log.append({"midi": str(midi_path), "output": str(output_path), "status": "succeeded"})
        except Exception as exc:  # noqa: BLE001 - pipeline logs and continues
            failed += 1
            log.append(
                {
                    "midi": str(midi_path),
                    "output": str(output_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    elapsed = time.perf_counter() - start
    return PipelineResult(
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        elapsed_seconds=elapsed,
        log=log,
    )


def _compute_duration(notes: Iterable[Dict[str, Any]], padding_sec: float) -> float:
    notes_list = list(notes)
    if not notes_list:
        return max(0.0, float(padding_sec))
    last = max(float(note["start_sec"]) + float(note["duration_sec"]) for note in notes_list)
    return max(0.0, last + float(padding_sec))


def _resolve_output_path(midi_path: Path, out_dir: Path, cfg: PipelineConfig) -> Path:
    ext = f".{cfg.io.output_format}"
    stem = midi_path.stem
    name = cfg.io.file_naming.format(stem=stem)
    return out_dir / f"{name}{ext}"


def _get_worker_count(cfg: PipelineConfig) -> int:
    return cfg.pipeline.max_workers