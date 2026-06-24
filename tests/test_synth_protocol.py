from pathlib import Path

import numpy as np
import pytest

from sonitra.config import RenderingMode, load_config
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
    for synth_cls in [DawDreamerSynth, PedalboardSynth]:
        synth = synth_cls(sample_rate=44100)
        notes = parse_midi(midi_fixture("test_c4.mid"))
        audio = synth.render(notes, duration_sec=2.0)
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 2


def test_synth_factory_returns_correct_type_dawdreamer(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pipeline.rendering_mode = RenderingMode.DAWDREAMER_ONLY
    synth = make_synth(cfg)
    assert isinstance(synth, DawDreamerSynth)


def test_synth_factory_returns_correct_type_pedalboard(config_fixture):
    cfg = load_config(config_fixture("config_pedalboard_only.yaml"))
    synth = make_synth(cfg)
    assert isinstance(synth, PedalboardSynth)


def test_synth_factory_hybrid_returns_dawdreamer(config_fixture):
    cfg = load_config(config_fixture("config_valid.yaml"))
    cfg.pipeline.rendering_mode = RenderingMode.DAWDREAMER_SYNTH_PEDALBOARD_FX
    synth = make_synth(cfg)
    assert isinstance(synth, DawDreamerSynth)


def test_synth_factory_returns_fluid_synth_when_soundfont_configured(config_fixture):
    cfg = load_config(config_fixture("config_soundfont.yaml"))
    synth = make_synth(cfg)
    assert isinstance(synth, FluidSynth)


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
