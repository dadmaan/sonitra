import json

from midi_renderer.effects.builtin_effects import CompressorConfig
from midi_renderer.effects.chain_builder import compute_chain_hash
from midi_renderer.manifest import ManifestEntry, ManifestWriter


def test_write_creates_manifest_file(tmp_path):
    writer = ManifestWriter(tmp_path / "renders.jsonl")
    writer.write(
        ManifestEntry(
            midi_path="corpus/a.mid",
            output_path="out/a.wav",
            rendering_mode="dawdreamer_synth_pedalboard_fx",
            effects_chain_hash="abc123",
            status="done",
            duration_sec=3.1,
            rms=-18.3,
            peak=-1.0,
            elapsed_seconds=0.42,
        )
    )
    assert (tmp_path / "renders.jsonl").exists()


def test_each_write_appends_one_line(tmp_path):
    writer = ManifestWriter(tmp_path / "renders.jsonl")
    for i in range(5):
        writer.write(
            ManifestEntry(
                midi_path=f"a{i}.mid",
                output_path=f"a{i}.wav",
                rendering_mode="pedalboard_only",
                effects_chain_hash="x",
                status="done",
                duration_sec=1.0,
                rms=-20.0,
                peak=-1.0,
                elapsed_seconds=0.1,
            )
        )
    lines = (tmp_path / "renders.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5


def test_manifest_line_is_valid_json(tmp_path):
    writer = ManifestWriter(tmp_path / "renders.jsonl")
    writer.write(
        ManifestEntry(
            midi_path="a.mid",
            output_path="a.wav",
            rendering_mode="pedalboard_only",
            effects_chain_hash="y",
            status="done",
            duration_sec=1.0,
            rms=-20.0,
            peak=-1.0,
            elapsed_seconds=0.1,
        )
    )
    line = (tmp_path / "renders.jsonl").read_text().strip()
    parsed = json.loads(line)
    assert parsed["midi_path"] == "a.mid"
    assert parsed["status"] == "done"


def test_effects_chain_hash_deterministic():
    cfg_a = [CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=100, enabled=True)]
    cfg_b = [CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=100, enabled=True)]
    assert compute_chain_hash(cfg_a) == compute_chain_hash(cfg_b)


def test_effects_chain_hash_changes_with_params():
    cfg_a = [CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=100, enabled=True)]
    cfg_b = [CompressorConfig(threshold_db=-6, ratio=4, attack_ms=5, release_ms=100, enabled=True)]
    assert compute_chain_hash(cfg_a) != compute_chain_hash(cfg_b)


def test_failed_list_written_on_failure(tmp_path):
    writer = ManifestWriter(tmp_path / "renders.jsonl", failed_list_path=tmp_path / "failed.txt")
    writer.write(
        ManifestEntry(
            midi_path="bad.mid",
            output_path="",
            rendering_mode="pedalboard_only",
            effects_chain_hash="z",
            status="failed",
            duration_sec=0.0,
            rms=0.0,
            peak=0.0,
            elapsed_seconds=0.0,
        )
    )
    assert (tmp_path / "failed.txt").exists()
    assert "bad.mid" in (tmp_path / "failed.txt").read_text()
