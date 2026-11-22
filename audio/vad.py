from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .types import AudioChunk, SpeechSegment


class SileroVadOnnx:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        started = time.perf_counter()
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def reset_states(self) -> None:
        self.h.fill(0)
        self.c.fill(0)

    def __call__(self, frame: np.ndarray) -> float:
        if len(frame) != 512:
            raise ValueError(f"Silero VAD frame 必须为 512 samples，实际 {len(frame)}")
        probability, self.h, self.c = self.session.run(
            None,
            {"x": frame.reshape(1, -1).astype(np.float32), "h": self.h, "c": self.c},
        )
        return float(probability.reshape(-1)[0])


class VadSegmenter:
    def __init__(
        self,
        predictor: Callable[[np.ndarray], float],
        sample_rate: int,
        frame_samples: int,
        config: dict[str, Any],
    ):
        if frame_samples != 512:
            raise ValueError("当前 Silero ONNX 权重要求 audio.frame_samples=512")
        self.predictor = predictor
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.threshold = float(config.get("threshold", 0.5))
        self.min_speech_samples = int(sample_rate * float(config.get("min_speech_ms", 250)) / 1000)
        self.min_silence_samples = int(sample_rate * float(config.get("min_silence_ms", 650)) / 1000)
        self.max_speech_samples = int(sample_rate * float(config.get("max_speech_sec", 15)))
        self.min_segment_samples = int(sample_rate * float(config.get("min_segment_ms", 500)) / 1000)
        self.pre_frames = max(1, math.ceil(sample_rate * float(config.get("pre_pad_ms", 200)) / 1000 / frame_samples))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_frames)
        self._active_frames: list[np.ndarray] = []
        self._active = False
        self._candidate_samples = 0
        self._candidate_start_sample = 0
        self._speech_start_sample = 0
        self._last_speech_end_sample = 0
        self._last_speech_end_time: float | None = None
        self._stream_sample = 0
        self._vad_compute_ms = 0.0
        self._segment_counter = 0
        self.discarded_short_segments = 0

    def _segment(self, reason: str, source_mode: str) -> SpeechSegment | None:
        valid_samples = max(0, self._last_speech_end_sample - self._speech_start_sample)
        if valid_samples < self.min_segment_samples or not self._active_frames:
            self.discarded_short_segments += 1
            result = None
        else:
            self._segment_counter += 1
            tail_samples = max(0, self._stream_sample - self._last_speech_end_sample)
            result = SpeechSegment(
                segment_id=f"{self._segment_counter:04d}",
                samples=np.ascontiguousarray(np.concatenate(self._active_frames), dtype=np.float32),
                sample_rate=self.sample_rate,
                valid_speech_sec=valid_samples / self.sample_rate,
                vad_compute_ms=self._vad_compute_ms,
                tail_wait_ms=tail_samples / self.sample_rate * 1000.0,
                end_reason=reason,
                last_speech_end_monotonic=self._last_speech_end_time if source_mode == "realtime" else None,
                vad_confirmed_monotonic=time.perf_counter(),
                source_mode=source_mode,  # type: ignore[arg-type]
            )
        retained = self._active_frames[-self.pre_frames:] if self._active_frames else []
        self._pre_roll = deque((frame.copy() for frame in retained), maxlen=self.pre_frames)
        self._active_frames = []
        self._active = False
        self._candidate_samples = 0
        self._vad_compute_ms = 0.0
        return result

    def process(self, chunk: AudioChunk) -> list[SpeechSegment]:
        if chunk.sample_rate != self.sample_rate:
            raise ValueError(f"VAD 采样率不匹配: {chunk.sample_rate}")
        outputs: list[SpeechSegment] = []
        for offset in range(0, len(chunk.samples), self.frame_samples):
            frame = chunk.samples[offset:offset + self.frame_samples]
            actual_length = len(frame)
            if actual_length < self.frame_samples:
                frame = np.pad(frame, (0, self.frame_samples - actual_length))
            frame = np.ascontiguousarray(frame, dtype=np.float32)
            frame_start_sample = self._stream_sample
            frame_end_sample = frame_start_sample + actual_length
            frame_time = (
                chunk.start_time_monotonic + offset / self.sample_rate
                if chunk.start_time_monotonic is not None else None
            )
            started = time.perf_counter()
            probability = self.predictor(frame)
            self._vad_compute_ms += (time.perf_counter() - started) * 1000.0
            is_speech = probability >= self.threshold

            if not self._active:
                self._pre_roll.append(frame.copy())
                if is_speech:
                    if self._candidate_samples == 0:
                        self._candidate_start_sample = frame_start_sample
                    self._candidate_samples += actual_length
                    if self._candidate_samples >= self.min_speech_samples:
                        self._active = True
                        self._active_frames = [item.copy() for item in self._pre_roll]
                        self._speech_start_sample = self._candidate_start_sample
                        self._last_speech_end_sample = frame_end_sample
                        self._last_speech_end_time = (
                            frame_time + actual_length / self.sample_rate if frame_time is not None else None
                        )
                else:
                    self._candidate_samples = 0
            else:
                self._active_frames.append(frame.copy())
                if is_speech:
                    self._last_speech_end_sample = frame_end_sample
                    self._last_speech_end_time = (
                        frame_time + actual_length / self.sample_rate if frame_time is not None else None
                    )

            self._stream_sample = frame_end_sample
            if self._active:
                active_samples = self._stream_sample - self._speech_start_sample
                silence_samples = self._stream_sample - self._last_speech_end_sample
                if active_samples >= self.max_speech_samples:
                    segment = self._segment("max_duration", chunk.source_mode)
                    if segment:
                        outputs.append(segment)
                elif silence_samples >= self.min_silence_samples:
                    segment = self._segment("silence", chunk.source_mode)
                    if segment:
                        outputs.append(segment)
        return outputs

    def flush(self, source_mode: str, reason: str = "audio_end") -> list[SpeechSegment]:
        if not self._active:
            return []
        segment = self._segment(reason, source_mode)
        return [segment] if segment else []

