import numpy as np

from midi_renderer.config import RenderingMode, load_config
from midi_renderer.midi_reader import parse_midi
from midi_renderer.synth.dawdreamer_synth import DawDreamerSynth
from midi_renderer.synth.pedalboard_synth import PedalboardSynth
from midi_renderer.synth.protocol import SynthesiserProtocol, make_synth


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
