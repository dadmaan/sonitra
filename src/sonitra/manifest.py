from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading


@dataclass
class ManifestEntry:
    midi_path: str
    output_path: str
    synth_backend: str
    effects_chain_hash: str
    status: str
    duration_sec: float
    rms: float
    peak: float
    elapsed_seconds: float
    quality_flags: dict | None = None
    source_path: str | None = None


class ManifestWriter:
    def __init__(self, manifest_path: Path | str, *, failed_list_path: Path | str | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.failed_list_path = Path(failed_list_path) if failed_list_path else None
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.failed_list_path:
            self.failed_list_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, entry: ManifestEntry) -> None:
        payload = asdict(entry)
        with self._lock:
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            if entry.status == "failed" and self.failed_list_path:
                with self.failed_list_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{entry.midi_path}\n")
