from __future__ import annotations

import logging
import queue
import threading
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
        self.queue: queue.Queue[tuple[np.ndarray, float, int]] = queue.Queue(maxsize=queue_frames)
        self.dropped_frames = 0
        self.dropped_samples = 0
        self.overflow_events = 0
        self.overflow_samples = 0
        self.max_queue_depth = 0
        self._pending_discontinuity_samples = 0
        self._reported_dropped_frames = 0
        self._reported_overflow_events = 0
        self._reported_queue_depth = 0
        self._stats_lock = threading.Lock()
        self._stream: Any = None
        self._closed = False

    def _callback(self, indata: np.ndarray, frames: int, _time_info: Any, status: Any) -> None:
        callback_ended = time.perf_counter()
        chunk_started = callback_ended - frames / self.sample_rate
        samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        with self._stats_lock:
            if bool(getattr(status, "input_overflow", False)):
                self.overflow_events += 1
                self.overflow_samples += frames
                self._pending_discontinuity_samples += frames
            try:
                self.queue.put_nowait((samples, chunk_started, self._pending_discontinuity_samples))
            except queue.Full:
                self.dropped_frames += 1
                self.dropped_samples += frames
                self._pending_discontinuity_samples += frames
            else:
                self._pending_discontinuity_samples = 0
                self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())

    def _warn_capture_loss(self) -> None:
        with self._stats_lock:
            dropped = self.dropped_frames
            overflows = self.overflow_events
            max_depth = self.max_queue_depth
            dropped_samples = self.dropped_samples
            overflow_samples = self.overflow_samples
        backlog_threshold = max(1, (self.queue.maxsize * 3 + 3) // 4)
        backlog_changed = max_depth >= backlog_threshold and max_depth > self._reported_queue_depth
        loss_changed = dropped != self._reported_dropped_frames or overflows != self._reported_overflow_events
        if loss_changed or backlog_changed:
            logging.getLogger("audio.capture").warning(
                "麦克风队列状态: depth=%s/%s max_depth=%s dropped_frames=%s "
                "dropped_samples=%s device_overflows=%s overflow_samples~=%s",
                self.queue.qsize(),
                self.queue.maxsize,
                max_depth,
                dropped,
                dropped_samples,
                overflows,
                overflow_samples,
            )
            self._reported_dropped_frames = dropped
            self._reported_overflow_events = overflows
            self._reported_queue_depth = max_depth

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "queue_depth": self.queue.qsize(),
                "queue_capacity": self.queue.maxsize,
                "max_queue_depth": self.max_queue_depth,
                "dropped_frames": self.dropped_frames,
                "dropped_samples": self.dropped_samples,
                "overflow_events": self.overflow_events,
                "overflow_samples_estimate": self.overflow_samples,
                "pending_discontinuity_samples": self._pending_discontinuity_samples,
            }

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
                samples, started, discontinuity_samples = self.queue.get(timeout=0.5)
            except queue.Empty:
                self._warn_capture_loss()
                continue
            self._warn_capture_loss()
            yield AudioChunk(
                samples,
                self.sample_rate,
                started,
                "realtime",
                discontinuity_samples=discontinuity_samples,
            )

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

