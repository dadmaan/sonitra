# Adding a transcription backend

This guide shows how Sonitra adds a real transcription backend. It uses TransKun as the worked example. Every new backend follows the same shape.

A transcriber turns audio into note dicts. A note dict has four keys: `pitch`, `velocity`, `start_sec`, `duration_sec`. All backends must return note dicts that meet the same contract.

## Config first

Add a config class in `src/sonitra/transcribe/configs.py`. It holds settings the user can set in YAML.

Sonitra validates config with Pydantic. Pydantic is a Python tool that checks settings. Each section uses `extra="forbid"`, so an unknown key stops the run with a clear error.

TransKun's config looks like this:

```python
class TranskunTranscriberConfig(_TranscriberBase):
    type: Literal["transkun"] = "transkun"
    device: str = "cpu"
    segment_size_sec: float | None = None
    segment_hop_sec: float | None = None
    weights_path: Path | str | None = None
    conf_path: Path | str | None = None
```

In YAML you enable it like this:

```yaml
transcription:
  transcribers:
    - type: transkun
      device: cpu                 # also accepts GPU:0, translated to cuda:0
      segment_size_sec: null      # null keeps the bundled conf value (16 s)
      segment_hop_sec: null       # null keeps the bundled conf value (8 s)
      weights_path: null          # null uses the bundled checkpoint
      conf_path: null            # null uses the bundled conf
```

`enabled` and `name` come from the base class. `device` accepts the unified strings `cpu`, `cuda`, `cuda:N` and `GPU:N` (plus `mps` on torch backends). Each backend translates them at its boundary with the shared helpers in `src/sonitra/transcribe/devices.py` (`resolve_torch_device` maps `GPU:N` to `cuda:N`; `resolve_tf_device` maps `cuda` to `GPU:0`), so users write the same word everywhere. Any new backend must accept the same strings and translate via that module; anything else raises `TranscriptionError` naming the valid values. A backend reads the process-level numeric settings (`transcription.numeric_mode`, `gpu_memory_growth`) in its builder, with `read_numeric_env()` from `src/sonitra/transcribe/numerics.py`. A run publishes the config's values into the two environment variables before it calls any builder, so the builder reads what the config asked for. Outside a run, `make_transcriber` reads whatever the environment holds, which is `off` and `false` when the variables are unset. A torch backend then applies those settings to the torch process through `src/sonitra/transcribe/torch_support.py` rather than on its own; see [Share the torch process settings](#share-the-torch-process-settings) below. Either way, record the effective `numeric_mode` in `metadata`. A backend that applies those settings raises `NumericSettingsError` (`src/sonitra/transcribe/base.py`) when it cannot apply one under `strict`, because every file after it would then run under settings that are not in force. Expose the application as `apply_numeric_settings()`, returning the fallbacks it hit (empty when everything applied) so a run can apply the settings before its first file instead of during it, and record those strings in `metadata` as `numeric_fallbacks`.

A registry entry needs **two** edits, not one: the config class in the `TranscriberConfig` union at the bottom of the file, and the module on the lazy import line in `protocol.py` (see [Register the backend](#register-the-backend)). Miss either and `tests/test_transcriber_registry.py` fails, because it compares the union against the modules that registered themselves. It imports each backend module by name to populate that list, so add your module there too, or the comparison fails with the new type missing from the registry side. Do all of it in one pass.

That union is a discriminated union. The `type` field picks which class to validate.

## Register the backend

Create `src/sonitra/transcribe/transkun.py` and register a builder:

```python
@register_transcriber("transkun")
def _build(cfg: TranskunTranscriberConfig) -> TranskunTranscriber:
    return TranskunTranscriber(
        device=cfg.device,
        segment_size_sec=cfg.segment_size_sec,
        segment_hop_sec=cfg.segment_hop_sec,
        weights_path=cfg.weights_path,
        conf_path=cfg.conf_path,
        name=cfg.name or "transkun",
    )
```

The string `"transkun"` is the `type` discriminator. Use it unchanged in config YAML. It must match the literal on the config class.

Then add the module to the lazy import line in `src/sonitra/transcribe/protocol.py` inside `make_transcriber`. That line runs on every `make_transcriber` call, even for `precomputed`. For that reason the backend module must not import heavy libraries at top level.

The line lists modules alphabetically, and `basic_pitch` stays first on it. Keep both properties when you insert a name:

```python
from sonitra.transcribe import basic_pitch, external_command, hft_transformer, precomputed, transkun  # noqa: F401
```

A reader scanning that line expects the first entry to be the always-present core backend; reordering it for cosmetic reasons breaks that expectation for no gain. Every module on the line must also be import-safe with its extra missing, because the whole line executes before the registry lookup decides which builder is needed.

Two names matter. `backend_type` is fixed. It holds the discriminator, for example `"transkun"`. It keys the writer registry. `transcriber` is the user visible name. It defaults to the backend type but the user can override it with `name` in YAML. Code that looks up a writer or result must use `backend_type`, not `transcriber`. A renamed instance would miss its own writer otherwise.

## Route every note through `make_note`

All backends must build notes with `make_note` from `src/sonitra/notes.py`. That function is the contract.

```python
from sonitra.notes import make_note

note = make_note(pitch=60, velocity=90, start_sec=1.0, duration_sec=0.5)
# Returns a dict or None
```

`make_note` does four things:

* `duration_sec <= 0` returns `None`. The caller drops the note. A real model can emit a zero length note. Dropping it is the correct handling.
* `velocity` is clamped to 1 to 127. MIDI velocity is loudness from 1 to 127.
* `start_sec < 0` is clamped to 0. Negative starts have no meaning for output.
* `pitch` outside 0 to 127 raises `ValueError`. Pitch is MIDI note number, where 60 is middle C and the piano spans 21 to 108. An out of range pitch means the calling code has a bug. A loud error is the point. It catches the pitch offset bug described below.

Sort the result by `(start_sec, pitch)` before you return it. Both Basic Pitch and TransKun do this.

### The pitch offset trap that code cannot catch

This is the most valuable trap to understand. TransKun assigns each pitch as `targetMIDIPitch = [-64, -67] + list(range(21, 108 + 1))`. The first two entries are pedal control values: -64 is sustain pedal, -67 is soft pedal. The rest are real MIDI numbers. A TransKun `Note.pitch` is already a correct MIDI number.

It is easy to think you must add 21, because many piano models map 0 to 52 to piano keys 21 to 108. If you add 21 here, an A0 at 21 becomes 42, and the top C at 108 becomes 129. Value 129 fails the 0 to 127 check, or if clamped, it silently shifts every note by a major sixth plus an octave. The output still looks plausible and scores. You get a valid but wrong result. No helper catches it, because 42 is inside 0 to 127.

Two rules prevent this. First, keep the pitch as is. Do not add 21. Second, filter pedal pitches before you call `make_note`. Filter any `pitch < 0` before the call. If you call `make_note` with -64, it raises. That raise is by design, but pedal events are not notes and should be dropped quietly. The tests that prove the mapping is correct are in `tests/test_transkun.py`:

* `test_notes_to_dicts_no_pitch_offset` checks that 21 stays 21 and 108 stays 108.
* `test_notes_to_dicts_filters_negative_pedal_before_make_note` checks that -64 and -67 are dropped without raising.

Boundary notes are kept. If a note has `hasOnset` false or `hasOffset` false, it means the note touches the edge of an audio segment. Sonitra keeps these notes. That matches upstream behaviour, so Sonitra numbers stay comparable to published results. The count of such notes is still recorded in metadata so you can see the effect.

## Use lazy imports and a clear error

Never import `torch` or `transkun` at the top of the module. Import inside `transcribe`. This keeps the core import light when the extra is not installed.

If the import fails, raise `TranscriptionError` with an install hint. On a torch backend, build it with the shared `missing_dependency_error` from `src/sonitra/transcribe/torch_support.py` instead of writing your own sentence:

```python
try:
    import torch
except ImportError as exc:
    raise missing_dependency_error("torch", backend="transkun") from exc
```

That helper names the missing module and gives the same install command for every backend, so a user who hits it once can act on it without reading each backend's source. Do the same for the `transkun` package, and for anything else your backend imports. The factory will only create this backend when the user config asks for `type: transkun`, so the error only appears when it is needed.

## Share the torch process settings

Torch's numeric flags and its device state are process-global, and two torch backends can be built in the same worker. So the settings live in one place: `src/sonitra/transcribe/torch_support.py`, with three public helpers a torch backend must use. The values themselves come from `read_numeric_env()` in `src/sonitra/transcribe/numerics.py`. Call that function in the builder, not in `transcribe`: the builder runs once per instance, and `transcribe` runs per file.

- `validate_torch_device(device, backend=...)` resolves a unified device string and confirms it exists. It returns `(resolved_name, available)` and raises `TranscriptionError` when an accelerator was requested and is absent. `cpu` short-circuits with no framework import at all.
- `apply_torch_numeric_settings(numeric_mode, gpu_memory_growth, backend=..., device=...)` applies the process-global torch state and returns the fallbacks it accepted, empty when the requested settings were applied in full. Pass the **resolved** device, because it decides the cuBLAS workspace below.
- `missing_dependency_error(module, backend=...)` builds the install-hint error described above.

Two rules make the sharing safe, and a backend that re-implements either of them breaks the other backend instead of helping.

**One cache, not one per backend.** The module keeps a single `_NUMERIC_STATE` entry pairing the `(numeric_mode, gpu_memory_growth)` this process has already applied with the fallbacks that took, and `apply_torch_numeric_settings` returns those fallbacks without touching torch when the pair matches. That is deliberate, and it is the whole reason the helpers are shared. The settings are process-global, so if each backend kept its own cache then two backends asking for different modes would each apply their own and whichever ran last would silently win for the entire process, including for the other backend's remaining files. There is no per-backend answer to that question, so it must not become one. A strict failure leaves the entry unset, so the next file asks again instead of being handed an unverified "already applied". Do not keep a second cache, and do not call `torch.use_deterministic_algorithms` yourself outside this helper.

**The cuBLAS workspace default is narrow on purpose.** Deterministic cuBLAS needs a fixed workspace size, so the helper sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` as an environment default. That fires only when `numeric_mode` is `strict` **and** the resolved device is a CUDA device. It is not applied for `warn`, and not on CPU, where cuBLAS is not in the path at all. `setdefault` is deliberate: an operator who exports their own value keeps it.

The rest of the behaviour is the same on both frameworks. `warn` maps to `torch.use_deterministic_algorithms(True, warn_only=True)` and falls back with a warning where the build lacks it, `strict` raises, and TF32 is off in both modes because it is deterministic but less accurate. `gpu_memory_growth` is TensorFlow-only; torch already grows its cache, so the argument is accepted for a uniform section schema and does nothing.

## Load audio at 44.1 kHz

TransKun requires 44.1 kHz. Sonitra provides `read_audio_resampled` in `src/sonitra/storage.py` for this.

```python
from sonitra.storage import read_audio_resampled

audio, sr = read_audio_resampled(audio_path, target_sr=44100)
```

`read_audio_resampled` uses `pedalboard.io.AudioFile(path).resampled_to(44100)`. Pedalboard is an audio library Sonitra already uses. This call handles wav, flac and mp3 without ffmpeg. TransKun's own CLI calls `pydub.AudioSegment.from_mp3` unconditionally, which needs ffmpeg. Sonitra avoids that by reusing pedalboard.

The function lives in `storage.py` rather than inside the transcriber so other backends can reuse it and it can be tested without torch. It also means the benchmark works when `io.output_format` is `mp3`. A file written as mp3 decodes longer than it was written, by about 64 ms in one measured case, so comparison code should not rely on exact frame counts for mp3.

TransKun's model expects audio as `(samples, channels)` after its own read path. `read_audio_resampled` returns `(channels, samples)`. The backend transposes if needed before building the torch tensor. Keep that conversion explicit and tested.

## Locate the checkpoint with `importlib.resources`

TransKun wheels bundle a checkpoint at `transkun/pretrained/2.0.pt` and a config at `pretrained/2.0.conf`. The TransKun CLI used `pkg_resources`, which is removed in newer setuptools. Use `importlib.resources`:

```python
import importlib.resources as resources

weight_path = Path(str(resources.files("transkun").joinpath("pretrained/2.0.pt")))
conf_file = Path(str(resources.files("transkun").joinpath("pretrained/2.0.conf")))
```

If the user set `weights_path` or `conf_path` in YAML, use those instead. Read the conf with `moduleconf.parseFromFile`, then apply `segment_size_sec` and `segment_hop_sec` overrides if they are not None. Passing `None` keeps the bundled conf value as the source of truth.

A backend whose weights are not bundled needs a different answer, and there are only two workable ones: ship them inside the package, or have the user install them once out of band and resolve the location at run time. `hft_transformer` takes the second route, because its training data (MAESTRO, CC BY-NC-SA 4.0) forbids redistribution. Its `scripts/setup_hft_transformer.py` writes `<models_dir>/hft_transformer/maestro/model.pt` next to a `manifest.json`, and `src/sonitra/transcribe/_hft/checkpoints.py` resolves the models directory: `SONITRA_MODELS_DIR` when set and non-empty, otherwise `~/.cache/sonitra/models`, with no repository-relative default because a checkout may be read-only or shared. When the weights are missing, the backend raises `TranscriptionError` naming the setup command and the directory it looked in; it must never download at transcribe time, because a benchmark run is a loop and a per-file network call is not recoverable.

## Load weights and run inference safely

We own the `torch.load` call, so we handle the flag:

```python
try:
    checkpoint = torch.load(str(weight_path), map_location=resolved, weights_only=True)
except TypeError:
    checkpoint = torch.load(str(weight_path), map_location=resolved)
```

The checkpoint holds state dicts plus ints. `weights_only=True` is safer when available. Older torch versions do not support the flag, so fall back.

Use a scoped `torch.no_grad()` block for inference. Do not call `torch.set_grad_enabled(False)`. That call latches a process global and stays false after the method returns. It would silently disable autograd for any other torch code in the same worker. `torch.no_grad()` only affects the block. A test in `tests/test_transkun.py` checks that grad mode is still enabled after `transcribe`.

Device handling must be strict, and it is shared. Call `validate_torch_device` from `src/sonitra/transcribe/torch_support.py` (`_resolve_device` in `transkun.py` is a thin wrapper over the shared torch helper, kept for backwards compatibility). It resolves the unified string and raises for you, so a backend does not re-derive the two failure messages:

```python
resolved, available = validate_torch_device(self.device, backend="transkun")
```

A cuda request with no CUDA must raise, not fall back to cpu, and the helper does exactly that. It also tells apart the two reasons a CUDA request fails, which are not the same problem: a CPU-only torch build (the fork was never swapped, see [devcontainer.md](devcontainer.md)) and a real CUDA build that cannot see a GPU (a passthrough problem). The benchmark records `transcribe_seconds` per cell. A silent fallback would label cpu timings as gpu results.

## Record provenance in `metadata`

Every `TranscriptionResult` carries a `metadata` dict. The benchmark copies it into `BenchmarkRecord.transcriber_metadata` for each row in `benchmark_results.jsonl`. Fill it with the standard keys:

* `package_version`: from `importlib.metadata.version("transkun")`, with fallback to sonitra version or `"unknown"`. Always present.
* `weights_sha256`: inside the same `checkpoint_identity` helper. Only present when the user set a custom `weights_path`. The helper hashes the file. On the default bundled path it does no I/O.
* `device`: the resolved device actually used, for example `cuda:0`.
* `requested_device`: the raw string from config, before translation.
* `device_available`: `True` for `cpu` with no framework import; otherwise whether the requested accelerator exists (GPU list / `cuda.is_available()` plus `cuda:N` index bound).
* `numeric_mode`: the effective process-level setting (`off`, `warn`, `strict`) applied in the lazy-import block.
* `numeric_fallbacks`: the settings that could not be applied, as a list, empty when everything applied or the mode is `off`. This is what tells a reader the row ran under weaker settings than the config asked for.
* `filtered_dropped`: notes the model returned that `make_note` dropped.
* `note_events_total` and `notes_kept`: before and after the filter.
* `boundary_incomplete`: notes with `hasOnset` or `hasOffset` false, counted but kept.

Those are the keys the other backends record, so a study can compare them. Add whatever else a reader would need to reproduce your run: `hft_transformer` records its decoder head, window stride, batch size, thresholds, resampler and sample rate, the upstream commit and checkpoint name, the weights' sha256, and two counts a reader needs to see the filter's effect (`filtered_dropped`, and `offset_past_end` for notes whose decoded end ran past the audio). A number that cannot be re-derived from the recorded keys is a number nobody can reproduce.

TransKun sets `raw_outputs` to `None`. It registers no raw writer, so `write_transcription_outputs` writes only MIDI.

## Thread safety

`sonitra transcribe` can run with `transcription.max_workers > 1`. It shares one transcriber instance across threads. Lazy init without a lock would let many threads each load the 56 MB checkpoint at once.

Guard both model creation and the inference call with a `threading.RLock`:

```python
self._lock = threading.RLock()

with self._lock:
    if self._model is None:
        self._model = load_model(...)
    notes_est = self._model.transcribe(x, ...)
```

Cache the model on `self._model`. That follows the `DemucsSeparator` pattern already in the codebase.

The benchmark is not affected in the same way. It rebuilds transcribers per condition and runs each condition in a separate process, so no sharing occurs.

## Licence and training data

Record the licence and training data for the model card, and the position on the weights. TransKun is MIT and ships its own checkpoint. hFT-Transformer's code is MIT but its weights are not redistributed, because they derive from MAESTRO under CC BY-NC-SA 4.0; a card has to say both halves, or a reader cannot tell what they may do with the result. The default TransKun checkpoint is trained on MAESTRO V3. Note any data overlap with Sonitra corpora in `docs/model-cards.md`, where the guide for that file asks for a note on MAESTRO overlap and one on input resolution across models (Basic Pitch resamples to 22.05 kHz, TransKun requires 44.1 kHz, hFT-Transformer's features are pinned to 16 kHz). State measured numbers with the corpus, the input mode, the device and the numeric mode that produced them, and say plainly when a comparison has not been run yet.

## Check your work

The acceptance gate is `tests/helpers.py::assert_notes_satisfy_contract`. It checks six things: required keys, pitch in 0 to 127, velocity in 1 to 127, start at least 0, duration positive, and sorted by `(start_sec, pitch)`. Use it in mapper tests while you can still run without the heavy extra, plus the slow inference tests once `uv sync --extra transkun` is installed.

Hand the helper a list that breaks each rule once and confirm it fails. Run `python -m pytest tests/ -m "not slow"` without transkun, then `uv run python -m pytest tests/test_transkun.py -m slow` with it. The existing `tests/test_transkun.py` already covers the pitch offset, pedal filtering, sorting, clamping, device translation and the grad mode check. Add similar coverage for any new backend.

---
[← Back to README](../README.md)
