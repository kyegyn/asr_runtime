from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from .types import AudioChunk


class MicrophoneCapture:
    """Non-blocking sounddevice callback feeding a bounded consumer queue."""

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        queue_frames: int,
        device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=queue_frames)
        self.dropped_frames = 0
        self._stream: Any = None
        self._closed = False

    def _callback(self, indata: np.ndarray, frames: int, _time_info: Any, status: Any) -> None:
        callback_ended = time.perf_counter()
        chunk_started = callback_ended - frames / self.sample_rate
        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        try:
            self.queue.put_nowait((samples, chunk_started))
        except queue.Full:
            self.dropped_frames += 1

    def __enter__(self) -> "MicrophoneCapture":
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __iter__(self) -> Iterator[AudioChunk]:
        while not self._closed:
            try:
                samples, started = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            yield AudioChunk(samples, self.sample_rate, started, "realtime")

    def close(self) -> None:
        self._closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __exit__(self, *_: object) -> None:
        self.close()


def microphone_devices() -> list[dict[str, Any]]:
    import sounddevice as sd

    devices = []
    for index, item in enumerate(sd.query_devices()):
        if int(item.get("max_input_channels", 0)) > 0:
            devices.append(
                {
                    "index": index,
                    "name": str(item.get("name", "")),
                    "max_input_channels": int(item.get("max_input_channels", 0)),
                    "default_samplerate": float(item.get("default_samplerate", 0)),
                }
            )
    return devices

