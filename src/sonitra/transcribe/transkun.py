"""TransKun piano transcription backend.

Requires the optional ``transkun`` extra
(``pip install sonitra[transkun]``). Model weights are bundled.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from sonitra.notes import make_note
from sonitra.transcribe.base import TranscriptionError, TranscriptionResult, checkpoint_identity
from sonitra.transcribe.configs import TranskunTranscriberConfig
from sonitra.transcribe.devices import resolve_torch_device
from sonitra.transcribe.protocol import register_transcriber
from sonitra.transcribe.torch_support import (
    apply_torch_numeric_settings,
    missing_dependency_error,
    numeric_settings_from_env,
    validate_torch_device,
)

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Map Basic Pitch ``GPU:*`` style strings to torch ``cuda:*``.

    Thin wrapper over :func:`sonitra.transcribe.devices.resolve_torch_device`
    kept so existing imports keep working; new code should use the shared
    helper directly.
    """
    return resolve_torch_device(device)


def _missing_dependency(module: str) -> TranscriptionError:
    """TranscriptionError naming the module that failed and how to install it."""
    return missing_dependency_error(module, backend="transkun")


def _notes_to_dicts(notes: Any) -> list[dict[str, Any]]:
    """Map transkun ``Note`` objects to Sonitra note dicts via ``make_note``."""
    out: list[dict[str, Any]] = []
    for n in notes:
        pitch = getattr(n, "pitch", None)
        if pitch is None:
            continue
        try:
            pitch_int = int(pitch)
        except Exception:
            continue
        if pitch_int < 0:
            # pedal CC events (-64 sustain, -67 soft) filtered before make_note
            continue
        # velocity may be missing; default 64 matches make_note default via normalise_notes
        velocity = getattr(n, "velocity", 64)
        start = getattr(n, "start", getattr(n, "start_sec", 0.0))
        end = getattr(n, "end", None)
        if end is not None:
            try:
                duration = float(end) - float(start)
            except Exception:
                continue
        else:
            try:
                duration = float(getattr(n, "duration_sec", 0.0))
            except Exception:
                continue
        try:
            note_dict = make_note(
                pitch=pitch_int,
                velocity=int(velocity) if velocity is not None else 64,
                start_sec=float(start),
                duration_sec=duration,
            )
        except ValueError:
            # pitch out of 0..127 is a bug — propagate loudly
            raise
        except Exception:
            continue
        if note_dict is not None:
            out.append(note_dict)
    out.sort(key=lambda d: (d["start_sec"], d["pitch"]))
    return out


def _transkun_package_version() -> str:
    try:
        from importlib.metadata import version

        try:
            return version("transkun")
        except Exception:
            try:
                return version("sonitra")
            except Exception:
                return "unknown"
    except Exception:
        return "unknown"


def _apply_numeric_settings(
    numeric_mode: str, gpu_memory_growth: bool, *, device: str | None = None
) -> None:
    """Apply process-global torch numeric settings for this backend."""
    apply_torch_numeric_settings(
        numeric_mode, gpu_memory_growth, backend="transkun", device=device
    )


class TranskunTranscriber:
    """TransKun piano transcription.

    Requires ``pip install sonitra[transkun]``.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        segment_size_sec: float | None = None,
        segment_hop_sec: float | None = None,
        weights_path: Path | str | None = None,
        conf_path: Path | str | None = None,
        numeric_mode: str = "off",
        gpu_memory_growth: bool = False,
        name: str = "transkun",
    ) -> None:
        self.device = device
        self.segment_size_sec = segment_size_sec
        self.segment_hop_sec = segment_hop_sec
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.conf_path = Path(conf_path) if conf_path is not None else None
        mode = numeric_mode.lower() if isinstance(numeric_mode, str) else numeric_mode
        if mode not in ("off", "warn", "strict"):
            raise TranscriptionError(
                f"transkun got unknown numeric_mode {numeric_mode!r}; "
                "use 'off', 'warn' or 'strict'."
            )
        self.numeric_mode = mode
        self.gpu_memory_growth = bool(gpu_memory_growth)
        self.name = name
        self._model: Any | None = None
        self._lock = threading.RLock()

    def validate_device(self) -> tuple[str, bool]:
        """Resolve the configured device and confirm it exists.

        Returns (resolved_name, available). Raises TranscriptionError when an
        accelerator was requested but is absent. CPU short-circuits with no
        framework import.
        """
        return validate_torch_device(self.device, backend="transkun")

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        # Fail fast on a missing accelerator; also the backstop for worker
        # subprocesses and direct library use.
        resolved, _available = self.validate_device()
        # lazy imports so torch is never touched unless this backend is used
        try:
            import torch
        except ImportError as exc:
            raise _missing_dependency("torch") from exc

        _apply_numeric_settings(self.numeric_mode, self.gpu_memory_growth, device=resolved)

        # lazy load of transkun package itself
        try:
            import importlib.resources as resources
            import moduleconf
        except ImportError as exc:
            raise _missing_dependency("moduleconf") from exc

        # import needed helpers lazily
        try:
            # verify transkun installed; ModelTransformer is loaded via moduleconf
            import transkun  # noqa: F401
        except ImportError as exc:
            raise _missing_dependency("transkun") from exc

        from sonitra.storage import read_audio_resampled

        audio_path_p = Path(audio_path)

        # thread-safe lazy init + inference
        with self._lock:
            if self._model is None:
                # locate bundled weights / conf via importlib.resources
                if self.weights_path is not None:
                    weight_path = Path(self.weights_path)
                else:
                    try:
                        weight_path = Path(str(resources.files("transkun").joinpath("pretrained/2.0.pt")))
                    except Exception as exc:
                        raise TranscriptionError(f"cannot locate bundled transkun weights: {exc}") from exc

                if self.conf_path is not None:
                    conf_file = Path(self.conf_path)
                else:
                    try:
                        conf_file = Path(str(resources.files("transkun").joinpath("pretrained/2.0.conf")))
                    except Exception as exc:
                        raise TranscriptionError(f"cannot locate bundled transkun conf: {exc}") from exc

                # config overrides
                conf_manager = moduleconf.parseFromFile(str(conf_file))
                transkun_cls = conf_manager["Model"].module.TransKun
                conf = conf_manager["Model"].config
                if self.segment_size_sec is not None:
                    conf.segmentSizeInSecond = float(self.segment_size_sec)
                if self.segment_hop_sec is not None:
                    conf.segmentHopSizeInSecond = float(self.segment_hop_sec)

                # torch.load with weights_only fallback
                try:
                    checkpoint = torch.load(str(weight_path), map_location=resolved, weights_only=True)
                except TypeError:
                    # older torch without weights_only
                    checkpoint = torch.load(str(weight_path), map_location=resolved)
                except Exception as exc:
                    raise TranscriptionError(f"failed to load transkun checkpoint: {exc}") from exc

                model = transkun_cls(conf=conf).to(resolved)
                # checkpoint may hold state_dict or best_state_dict
                if "best_state_dict" in checkpoint:
                    state = checkpoint["best_state_dict"]
                elif "state_dict" in checkpoint:
                    state = checkpoint["state_dict"]
                else:
                    state = checkpoint
                try:
                    model.load_state_dict(state, strict=False)
                except Exception as exc:
                    raise TranscriptionError(f"failed to load transkun state dict: {exc}") from exc
                model.eval()
                self._model = model
            else:
                model = self._model

            # audio loading: always resample to 44100, handles wav/flac/mp3 via pedalboard
            try:
                audio, sr = read_audio_resampled(audio_path_p, target_sr=44100)
            except Exception as exc:
                raise TranscriptionError(f"failed to read audio {audio_path_p}: {exc}") from exc

            # model expects (samples, channels) or (channels, samples)? transkun's CLI uses
            # x = torch.from_numpy(audio).to(device) where audio shape is (frames, channels)
            # after readAudio with pydub. Our read_audio_resampled returns (channels, samples).
            # Convert to (samples, channels) for compatibility with model.transcribe which does x.transpose.
            # The model then does x.transpose(-1,-2) and pads etc.
            # Provide as (samples, channels) by transposing if needed.
            # read_audio_resampled gives (channels, samples); transpose to (samples, channels)
            if audio.ndim == 2:
                audio_for_model = audio.T  # (samples, channels)
            elif audio.ndim == 1:
                audio_for_model = audio[:, None]
            else:
                audio_for_model = audio

            x = torch.from_numpy(audio_for_model).to(resolved)

            # inference scoped no_grad
            with torch.no_grad():
                try:
                    notes_est = model.transcribe(
                        x,
                        stepInSecond=self.segment_hop_sec if self.segment_hop_sec is not None else None,
                        segmentSizeInSecond=self.segment_size_sec if self.segment_size_sec is not None else None,
                    )
                except Exception as exc:
                    raise TranscriptionError(f"transkun inference failed: {exc}") from exc

        # map to dicts, collect provenance
        raw_count = len(notes_est) if notes_est is not None else 0
        notes = _notes_to_dicts(notes_est if notes_est is not None else [])

        pkg_version = _transkun_package_version()
        weights_ident = self.weights_path if self.weights_path is not None else None
        checkpoint = checkpoint_identity(pkg_version, weights_ident)

        filtered_dropped = raw_count - len(notes)
        # count boundary notes kept (hasOnset/hasOffset false) — upstream parity keeps them
        boundary_incomplete = 0
        try:
            for n in notes_est or []:
                if not getattr(n, "hasOnset", True) or not getattr(n, "hasOffset", True):
                    boundary_incomplete += 1
        except Exception:
            boundary_incomplete = 0

        # device_available is what validate_device() reported (True here —
        # a missing accelerator raises before any metadata is built).
        metadata: dict[str, Any] = {
            **checkpoint,
            "device": resolved,
            "requested_device": self.device,
            "device_available": _available,
            "numeric_mode": self.numeric_mode,
            "filtered_dropped": filtered_dropped,
            "note_events_total": raw_count,
            "notes_kept": len(notes),
            "boundary_incomplete": boundary_incomplete,
        }

        return TranscriptionResult(
            notes=notes,
            transcriber=self.name,
            source_audio=audio_path_p,
            raw_outputs=None,
            backend_type="transkun",
            metadata=metadata,
        )


@register_transcriber("transkun")
def _build(cfg: TranskunTranscriberConfig) -> TranskunTranscriber:
    numeric_mode, gpu_memory_growth = numeric_settings_from_env()
    return TranskunTranscriber(
        device=cfg.device,
        segment_size_sec=cfg.segment_size_sec,
        segment_hop_sec=cfg.segment_hop_sec,
        weights_path=cfg.weights_path,
        conf_path=cfg.conf_path,
        numeric_mode=numeric_mode,
        gpu_memory_growth=gpu_memory_growth,
        name=cfg.name or "transkun",
    )
