from pathlib import Path

from midi_renderer.pipeline import run_pipeline


def test_pipeline_processes_single_midi(tmp_path, midi_fixture, session_engine):
    result = run_pipeline(
        midi_paths=[midi_fixture("test_c4.mid")],
        out_dir=tmp_path,
        engine=session_engine,
    )
    assert result.succeeded == 1
    assert result.failed == 0


def test_pipeline_output_file_exists(tmp_path, midi_fixture, session_engine):
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    assert len(list(tmp_path.glob("*.wav"))) == 1


def test_pipeline_batch_multiple_midis(tmp_path, midi_fixture, session_engine):
    midis = [midi_fixture("test_c4.mid"), midi_fixture("test_polyphonic.mid")]
    result = run_pipeline(midis, tmp_path, session_engine)
    assert result.succeeded == len(midis)


def test_pipeline_skips_existing_output(tmp_path, midi_fixture, session_engine):
    run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine, overwrite=False)
    assert result.skipped == 1


def test_pipeline_logs_failure_without_abort(tmp_path, session_engine, monkeypatch):
    def broken_render(*_args, **_kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr("midi_renderer.renderer.render_notes_faust", broken_render)
    result = run_pipeline([Path("nonexistent.mid")], tmp_path, session_engine)
    assert result.failed == 1
    assert result.succeeded == 0


def test_pipeline_result_contains_timing(tmp_path, midi_fixture, session_engine):
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    assert result.elapsed_seconds > 0.0


def test_pipeline_result_is_serialisable(tmp_path, midi_fixture, session_engine):
    result = run_pipeline([midi_fixture("test_c4.mid")], tmp_path, session_engine)
    import json

    json.dumps(result.to_dict())
