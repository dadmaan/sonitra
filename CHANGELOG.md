# Changelog

All notable changes to the sonitra project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pydantic config schema with YAML loader, rendering mode enum, and worker constraint validation
- Effects chain builder mapping 8 built-in pedalboard types plus VST3 plugins
- PedalboardSynth for MIDI-to-audio rendering via pedalboard instrument plugins
- SynthesiserProtocol with DawDreamerSynth wrapper and make_synth factory
- Peak/RMS normaliser with configurable pre/post-effects ordering
- Quality gates for silence, clipping, and minimum duration checks
- Manifest writer with JSONL render log, failed file list, and effects chain hash
- Config-driven pipeline wiring with three rendering modes
- FastAPI server with job queue, runtime config reload, and SSE status streaming
- Core audio engine, MIDI reader, Faust/VST renderer, and multi-format storage