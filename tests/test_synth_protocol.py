from pathlib import Path

import numpy as np
import pytest

from sonitra.config import SynthBackend, load_config
from sonitra.midi_reader import parse_midi
from sonitra.synth.dawdreamer_synth import DawDreamerSynth
from sonitra.synth.fluid_synth import FluidSynth
from sonitra.synth.pedalboard_synth import PedalboardSynth
from sonitra.synth.protocol import SynthesiserProtocol, make_synth

_DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/default-GM.sf2")
_SOUNDFONT_AVAILABLE = _DEFAULT_SOUNDFONT.exists() and _DEFAULT_SOUNDFONT.is_file()


def test_dawdreamer_synth_implements_protocol():
    synth = DawDreamerSynth(sample_rate=44100, block_size=512)
    assert isinstance(synth, SynthesiserProtocol)


def test_pedalboard_synth_implements_protocol():
    synth = PedalboardSynth(sample_rate=44100, channels=2)
    assert isinstance(synth, SynthesiserProtocol)


def test_protocol_render_signature_matches(midi_fixture):
    # DawDreamerSynth covers protocol shape without needing a plugin.
    synth = DawDreamerSynth(sample_rate=44100)
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=2.0)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 2


def test_pedalboard_synth_without_plugin_raises_on_render(midi_fixture):
    synth = PedalboardSynth(sample_rate=44100, channels=2, plugin_path=None)
    notes = parse_midi(midi_fixture("test_c4.mid"))
    with pytest.raises(ValueError, match="requires a VST instrument"):
        synth.render(notes, duration_sec=2.0)


def test_synth_factory_returns_correct_type_dawdreamer(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    # config_valid.yaml uses synth_backend: dawdreamer_faust
    assert cfg.pipeline.synth_backend == SynthBackend.DAWDREAMER_FAUST
    synth = make_synth(cfg)
    assert isinstance(synth, DawDreamerSynth)


def test_synth_factory_returns_correct_type_pedalboard(config_fixture):
    cfg = load_config(config_fixture("config_pedalboard_only.yaml"))
    synth = make_synth(cfg)
    assert isinstance(synth, PedalboardSynth)


def test_synth_factory_hybrid_returns_dawdreamer(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    # dawdreamer_faust uses DawDreamerSynth
    synth = make_synth(cfg)
    assert isinstance(synth, DawDreamerSynth)


def test_synth_factory_returns_fluid_synth_when_soundfont_configured(tmp_path) -> None:
    dummy_sf2 = tmp_path / "dummy.sf2"
    dummy_sf2.touch()
    from sonitra.config import PipelineConfig
    cfg = PipelineConfig.model_validate({
        "pipeline": {
            "synth_backend": "fluidsynth", "effects_chain": "none",
            "sample_rate": 44100, "bit_depth": 24, "channels": 2,
            "duration_padding_sec": 2.0, "overwrite": False, "resume": True,
            "max_workers": 1, "log_level": "INFO",
        },
        "io": {"corpus_root": ".", "output_format": "wav",
               "mp3_bitrate_kbps": 192, "file_naming": "{stem}"},
        "fluidsynth": {"soundfont_path": str(dummy_sf2)},
    })
    assert isinstance(make_synth(cfg), FluidSynth)


def test_synth_factory_returns_dawdreamer_with_vital(config_fixture):
    cfg = load_config(config_fixture("config_dawdreamer_vital.yaml"))
    synth = make_synth(cfg)
    assert isinstance(synth, DawDreamerSynth)


def test_synth_factory_returns_dawdreamer_with_vital_preset(config_fixture):
    cfg = load_config(config_fixture("config_dawdreamer_vital_preset.yaml"))
    synth = make_synth(cfg)
    assert isinstance(synth, DawDreamerSynth)
    assert synth.preset_path is not None


@pytest.mark.skipif(
    not _SOUNDFONT_AVAILABLE,
    reason="Default system SoundFont is not installed",
)
def test_fluid_synth_protocol_render_signature_matches(midi_fixture):
    synth = FluidSynth(
        sample_rate=44100,
        channels=2,
        soundfont_path=_DEFAULT_SOUNDFONT,
    )
    notes = parse_midi(midi_fixture("test_c4.mid"))
    audio = synth.render(notes, duration_sec=2.0)
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 2
