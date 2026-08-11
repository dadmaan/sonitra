from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Iterable, List
import time

import numpy as np
import pedalboard

_thread_local: threading.local = threading.local()

from sonitra.config import EffectsChain, PipelineConfig, SynthBackend
from sonitra.engine import RendererEngine
from sonitra.effects.chain_builder import build_effects_chain_from_config, compute_chain_hash
from sonitra.manifest import ManifestEntry, ManifestWriter
from sonitra.midi_reader import parse_midi
from sonitra.normaliser import normalise_from_config
from sonitra.quality_gate import check_quality
from sonitra.renderer import render_notes_faust, render_notes_vst
from sonitra.storage import derive_output_path, write_audio, write_wav
from sonitra.synth.protocol import make_synth


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


def _init_thread_synth_chain(cfg: PipelineConfig) -> None:
    """Create per-thread synth and effects-chain instances (called once per thread)."""
    _thread_local.synth = make_synth(cfg)
    _thread_local.chain = build_effects_chain_from_config(cfg)


def _render_file(
    midi_path: Path,
    out_dir: Path,
    cfg: PipelineConfig,
    chain_hash: str,
    manifest: ManifestWriter | None,
    corpus_root: Path | None,
) -> Dict[str, Any]:
    """Render one MIDI file using the calling thread's synth/chain.

    Returns a log-entry dict with a ``'status'`` key: ``'succeeded'``,
    ``'failed'``, or ``'skipped'``.  Writes to *manifest* directly (which is
    thread-safe after Tier-0 changes).
    """
    output_path = _resolve_output_path(midi_path, out_dir, cfg, corpus_root)
    if output_path.exists() and not cfg.pipeline.overwrite:
        return {"midi": str(midi_path), "output": str(output_path), "status": "skipped"}

    synth = _thread_local.synth
    chain = _thread_local.chain

    try:
        _meta = parse_midi(midi_path, return_meta=True)
        notes: List[Dict[str, Any]] = _meta["notes"]
        native_bpm: float = _meta["bpm"]
        if native_bpm > 0:
            notes = _scale_note_timings(notes, native_bpm / cfg.pipeline.bpm)
        duration = _compute_duration(notes, cfg.pipeline.duration_padding_sec)
        audio = synth.render(notes, duration_sec=duration)
        audio = normalise_from_config(audio, cfg, stage="pre")
        if cfg.pipeline.effects_chain == EffectsChain.PEDALBOARD and len(chain) > 0:
            audio = chain(audio, cfg.pipeline.sample_rate)
        audio = normalise_from_config(audio, cfg, stage="post")

        quality = check_quality(audio, cfg.pipeline.sample_rate, cfg.quality_gates)
        if not quality.passed:
            entry = ManifestEntry(
                midi_path=str(midi_path),
                output_path=str(output_path),
                synth_backend=cfg.pipeline.synth_backend.value,
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
            return {
                "midi": str(midi_path),
                "output": str(output_path),
                "status": "failed",
                "quality_flags": quality.to_dict(),
            }

        write_audio(
            audio,
            output_path,
            sample_rate=cfg.pipeline.sample_rate,
            bit_depth=cfg.pipeline.bit_depth,
            output_format=cfg.io.output_format,
            mp3_bitrate_kbps=cfg.io.mp3_bitrate_kbps,
            overwrite=cfg.pipeline.overwrite,
        )
        entry = ManifestEntry(
            midi_path=str(midi_path),
            output_path=str(output_path),
            synth_backend=cfg.pipeline.synth_backend.value,
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
        return {
            "midi": str(midi_path),
            "output": str(output_path),
            "status": "succeeded",
            "quality_flags": quality.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - pipeline logs and continues
        if manifest:
            manifest.write(
                ManifestEntry(
                    midi_path=str(midi_path),
                    output_path=str(output_path),
                    synth_backend=cfg.pipeline.synth_backend.value,
                    effects_chain_hash=chain_hash,
                    status="failed",
                    duration_sec=0.0,
                    rms=0.0,
                    peak=0.0,
                    elapsed_seconds=0.0,
                    quality_flags={"error": str(exc)},
                )
            )
        return {
            "midi": str(midi_path),
            "output": str(output_path),
            "status": "failed",
            "error": str(exc),
        }


def run_pipeline(
    midi_paths: Iterable[Path | str],
    out_dir: Path | str,
    engine: RendererEngine | None = None,
    plugin_path: Path | str | None = None,
    *,
    overwrite: bool = True,
    config: PipelineConfig | None = None,
    corpus_root: Path | None = None,
    on_file_done: Callable[[Dict[str, Any]], None] | None = None,
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
        n_workers = _get_worker_count(cfg)
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

        _init_thread_synth_chain(cfg)  # synth + chain for the calling thread

        def _tally(entry: Dict[str, Any]) -> None:
            nonlocal succeeded, failed, skipped
            log.append(entry)
            s = entry["status"]
            if s == "succeeded":
                succeeded += 1
            elif s == "skipped":
                skipped += 1
            else:
                failed += 1
            if on_file_done is not None:
                on_file_done(entry)

        if cfg.pipeline.synth_backend == SynthBackend.PEDALBOARD_INSTRUMENT and n_workers > 1:
            with ThreadPoolExecutor(
                max_workers=n_workers,
                initializer=_init_thread_synth_chain,
                initargs=(cfg,),
            ) as executor:
                futures = [
                    executor.submit(
                        _render_file, Path(mp), out_dir, cfg, chain_hash, manifest, corpus_root
                    )
                    for mp in midi_paths
                ]
                for future in as_completed(futures):
                    _tally(future.result())
        else:
            for midi_path in midi_paths:
                _tally(_render_file(Path(midi_path), out_dir, cfg, chain_hash, manifest, corpus_root))

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


def _scale_note_timings(
    notes: List[Dict[str, Any]], scale: float
) -> List[Dict[str, Any]]:
    if abs(scale - 1.0) <= 1e-6:
        return notes
    return [
        {**n, "start_sec": n["start_sec"] * scale, "duration_sec": n["duration_sec"] * scale}
        for n in notes
    ]


def _compute_duration(notes: Iterable[Dict[str, Any]], padding_sec: float) -> float:
    notes_list = list(notes)
    if not notes_list:
        return max(0.0, float(padding_sec))
    last = max(float(note["start_sec"]) + float(note["duration_sec"]) for note in notes_list)
    return max(0.0, last + float(padding_sec))


def _resolve_output_path(midi_path: Path, out_dir: Path, cfg: PipelineConfig, corpus_root: Path | None = None) -> Path:
    ext = f".{cfg.io.output_format}"
    stem = midi_path.stem
    name = cfg.io.file_naming.format(stem=stem)
    if corpus_root is not None:
        rel = midi_path.relative_to(corpus_root)
        return out_dir / rel.parent / f"{name}{ext}"
    return out_dir / f"{name}{ext}"


def _get_worker_count(cfg: PipelineConfig) -> int:
    return cfg.pipeline.max_workers