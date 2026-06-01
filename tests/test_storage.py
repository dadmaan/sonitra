from pathlib import Path

import pytest
from scipy.io import wavfile

from sonitra.storage import derive_output_path, write_wav


def test_write_wav_creates_file(tmp_path, dummy_audio):
    out = write_wav(dummy_audio, tmp_path / "out.wav", sample_rate=44100)
    assert out.exists()


def test_written_wav_is_readable(tmp_path, dummy_audio):
    out = write_wav(dummy_audio, tmp_path / "out.wav", sample_rate=44100)
    sr, data = wavfile.read(out)
    assert sr == 44100
    assert data.shape[0] > 0


def test_output_duration_matches_input(tmp_path, dummy_audio):
    out = write_wav(dummy_audio, tmp_path / "out.wav", sample_rate=44100)
    sr, data = wavfile.read(out)
    assert abs(data.shape[0] - dummy_audio.shape[1]) < 2


def test_derive_output_path_preserves_stem(tmp_path):
    midi = Path("corpus/track_042.mid")
    out = derive_output_path(midi, out_dir=tmp_path, ext=".wav")
    assert out.stem == "track_042"
    assert out.suffix == ".wav"
    assert out.parent == tmp_path


def test_derive_output_path_does_not_collide():
    paths = [derive_output_path(Path(f"song_{i}.mid"), out_dir=Path("/out"), ext=".wav") for i in range(100)]
    assert len(set(paths)) == 100


def test_overwrite_false_raises_if_exists(tmp_path, dummy_audio):
    out = tmp_path / "out.wav"
    write_wav(dummy_audio, out, sample_rate=44100)
    with pytest.raises(FileExistsError):
        write_wav(dummy_audio, out, sample_rate=44100, overwrite=False)
