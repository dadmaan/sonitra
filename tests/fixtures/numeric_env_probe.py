"""External-command transcription stub recording the numeric mode it inherited."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    probe_path = os.environ.get("SONITRA_NUMERIC_ENV_PROBE")
    if probe_path:
        with open(probe_path, "a", encoding="utf-8") as handle:
            handle.write(f"{os.environ.get('SONITRA_NUMERIC_MODE')}\n")
    # argv[1] is the rendered audio the parent produced; only argv[2] is written.
    shutil.copy(Path(__file__).with_name("test_c4.mid"), sys.argv[2])


if __name__ == "__main__":
    main()