from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import AudioChunk, SpeechSegment


@dataclass(frozen=True)
class _VadFrame:
    samples: np.ndarray
    actual_length: int
    start_sample: int
    end_sample: int
    start_time_monotonic: float | None
    is_speech: bool
    compute_ms: float


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
        activation_frames = max(1, math.ceil(self.min_speech_samples / frame_samples))
        self._history_capacity = self.pre_frames + activation_frames
        self._history: deque[_VadFrame] = deque(maxlen=self._history_capacity)
        self._active_frames: list[_VadFrame] = []
        self._active = False
        self._candidate_samples = 0
        self._candidate_start_sample = 0
        self._speech_start_sample = 0
        self._last_speech_end_sample = 0
        self._last_speech_end_time: float | None = None
        self._speech_samples = 0
        self._stream_sample = 0
        self._segment_vad_compute_ms = 0.0
        self._total_vad_compute_ms = 0.0
        self._total_vad_frames = 0
        self._segment_counter = 0
        self.discarded_short_segments = 0

    def _reset_segment_state(self, retain_history: bool) -> None:
        retained = self._active_frames[-self.pre_frames:] if retain_history else []
        self._history = deque(retained, maxlen=self._history_capacity)
        self._active_frames = []
        self._active = False
        self._candidate_samples = 0
        self._speech_samples = 0
        self._segment_vad_compute_ms = 0.0

    def _segment(self, reason: str, source_mode: str, retain_history: bool = True) -> SpeechSegment | None:
        if self._speech_samples < self.min_segment_samples or not self._active_frames:
            self.discarded_short_segments += 1
            result = None
        else:
            self._segment_counter += 1
            tail_samples = max(0, self._stream_sample - self._last_speech_end_sample)
            audio_samples = np.concatenate(
                [frame.samples[:frame.actual_length] for frame in self._active_frames]
            )
            audio_sample_count = sum(frame.actual_length for frame in self._active_frames)
            speech_span_samples = max(0, self._last_speech_end_sample - self._speech_start_sample)
            result = SpeechSegment(
                segment_id=f"{self._segment_counter:04d}",
                samples=np.ascontiguousarray(audio_samples, dtype=np.float32),
                sample_rate=self.sample_rate,
                audio_start_monotonic=(
                    self._active_frames[0].start_time_monotonic if source_mode == "realtime" else None
                ),
                audio_duration_sec=audio_sample_count / self.sample_rate,
                speech_frame_sec=self._speech_samples / self.sample_rate,
                speech_span_sec=speech_span_samples / self.sample_rate,
                vad_frame_avg_ms=(
                    self._total_vad_compute_ms / self._total_vad_frames if self._total_vad_frames else 0.0
                ),
                vad_segment_compute_ms=self._segment_vad_compute_ms,
                tail_wait_ms=tail_samples / self.sample_rate * 1000.0,
                end_reason=reason,
                last_speech_end_monotonic=self._last_speech_end_time if source_mode == "realtime" else None,
                vad_confirmed_monotonic=time.perf_counter(),
                source_mode=source_mode,  # type: ignore[arg-type]
            )
        self._reset_segment_state(retain_history)
        return result

    def _handle_discontinuity(self, missing_samples: int, source_mode: str) -> list[SpeechSegment]:
        outputs: list[SpeechSegment] = []
        if self._active:
            segment = self._segment("input_gap", source_mode, retain_history=False)
            if segment:
                outputs.append(segment)
        else:
            self._active_frames = []
            self._history.clear()
            self._candidate_samples = 0
        self._stream_sample += max(0, missing_samples)
        reset_states = getattr(self.predictor, "reset_states", None)
        if callable(reset_states):
            reset_states()
        return outputs

    def process(self, chunk: AudioChunk) -> list[SpeechSegment]:
        if chunk.sample_rate != self.sample_rate:
            raise ValueError(f"VAD 采样率不匹配: {chunk.sample_rate}")
        outputs = self._handle_discontinuity(chunk.discontinuity_samples, chunk.source_mode) if chunk.discontinuity_samples else []
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
            compute_ms = (time.perf_counter() - started) * 1000.0
            self._total_vad_compute_ms += compute_ms
            self._total_vad_frames += 1
            is_speech = probability >= self.threshold
            vad_frame = _VadFrame(
                frame,
                actual_length,
                frame_start_sample,
                frame_end_sample,
                frame_time,
                is_speech,
                compute_ms,
            )

            if not self._active:
                self._history.append(vad_frame)
                if is_speech:
                    if self._candidate_samples == 0:
                        self._candidate_start_sample = frame_start_sample
                    self._candidate_samples += actual_length
                    if self._candidate_samples >= self.min_speech_samples:
                        self._active = True
                        self._active_frames = list(self._history)
                        self._speech_start_sample = self._candidate_start_sample
                        self._last_speech_end_sample = frame_end_sample
                        self._speech_samples = self._candidate_samples
                        self._segment_vad_compute_ms = sum(
                            item.compute_ms for item in self._active_frames
                        )
                        self._last_speech_end_time = (
                            frame_time + actual_length / self.sample_rate if frame_time is not None else None
                        )
                else:
                    self._candidate_samples = 0
            else:
                self._active_frames.append(vad_frame)
                self._segment_vad_compute_ms += compute_ms
                if is_speech:
                    self._speech_samples += actual_length
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

