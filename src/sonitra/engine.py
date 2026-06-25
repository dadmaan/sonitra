from __future__ import annotations

from pathlib import Path
from typing import Optional
import threading


class RendererEngine:
    def __init__(self, sample_rate: int, block_size: int) -> None:
        # Lazy import: dawdreamer is an optional heavy backend with a native
        # extension that requires libGL.  Importing it at module level would
        # force the shared library to be resolved whenever any API router is
        # imported, even when the DawDreamer backend is not configured.
        import dawdreamer as daw  # noqa: PLC0415

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.sample_rate = sample_rate
        self.block_size = block_size
        self._engine = daw.RenderEngine(float(sample_rate), int(block_size))
        self._thread_id = threading.get_ident()
        self._faust_processor = None
        self._faust_freq_param = None

    @property
    def engine(self) -> daw.RenderEngine:
        return self._engine

    def assert_thread(self) -> None:
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("Rendering must occur on the engine's creation thread.")

    def load_plugin(self, plugin_path: Path | str):
        self.assert_thread()
        path = Path(plugin_path)
        if not path.exists():
            raise FileNotFoundError(f"VST plugin not found: {path}")
        processor = self._engine.make_plugin_processor(path.stem, str(path))
        self._engine.load_graph([(processor, [])])
        return processor

    def render_faust_sine(
        self,
        *,
        freq: float = 440.0,
        duration_sec: Optional[float] = None,
        beats: Optional[float] = None,
        bpm: Optional[float] = None,
    ):
        self.assert_thread()
        if duration_sec is None:
            if beats is None or bpm is None or bpm <= 0:
                raise ValueError("beats and bpm are required when duration_sec is None")
            duration_sec = float(beats) * (60.0 / float(bpm))
        self._engine.set_bpm(float(bpm) if bpm else 120.0)

        if self._faust_processor is None:
            processor = self._engine.make_faust_processor("faust_sine")
            dsp = "freq = hslider(\"freq\", 440, 20, 2000, 1); process = os.osc(freq), os.osc(freq);"
            processor.set_dsp_string(dsp)
            if not processor.compile():
                raise RuntimeError("Failed to compile Faust oscillator.")
            params = processor.get_parameters_description()
            if not params:
                raise RuntimeError("Faust oscillator parameters not available.")
            self._faust_freq_param = params[0]["name"]
            self._faust_processor = processor

        self._faust_processor.set_parameter(self._faust_freq_param, float(freq))
        self._engine.load_graph([(self._faust_processor, [])])

        self._engine.render(float(duration_sec))
        return self._engine.get_audio()