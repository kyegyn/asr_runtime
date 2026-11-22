import numpy as np
import pytest

from asr_runtime.audio.types import AudioChunk
from asr_runtime.audio.vad import VadSegmenter
from asr_runtime.audio import vad as vad_module


class FakePredictor:
    def __init__(self, probabilities: list[float]):
        self.values = iter(probabilities)
        self.reset_count = 0

    def __call__(self, _frame: np.ndarray) -> float:
        return next(self.values)

    def reset_states(self) -> None:
        self.reset_count += 1


def test_silence_endpoint_and_capture_timestamp() -> None:
    segmenter = VadSegmenter(
        FakePredictor([0.9, 0.9, 0.1, 0.1]),
        16_000,
        512,
        {
            "threshold": 0.5,
            "min_speech_ms": 32,
            "min_silence_ms": 64,
            "max_speech_sec": 10,
            "pre_pad_ms": 32,
            "min_segment_ms": 32,
        },
    )
    outputs = []
    for index in range(4):
        outputs.extend(
            segmenter.process(
                AudioChunk(np.ones(512, dtype=np.float32), 16_000, 100.0 + index * 0.032, "realtime")
            )
        )
    assert len(outputs) == 1
    segment = outputs[0]
    assert segment.end_reason == "silence"
    assert segment.speech_frame_sec == 0.064
    assert segment.speech_span_sec == 0.064
    assert segment.audio_duration_sec == 0.128
    assert segment.audio_start_monotonic == 100.0
    assert segment.tail_wait_ms == 64.0
    assert abs((segment.last_speech_end_monotonic or 0) - 100.064) < 1e-6


def test_fast_mode_has_no_realtime_end_timestamp() -> None:
    segmenter = VadSegmenter(
        FakePredictor([0.9]),
        16_000,
        512,
        {"min_speech_ms": 1, "min_silence_ms": 100, "min_segment_ms": 1},
    )
    segmenter.process(AudioChunk(np.ones(512, dtype=np.float32), 16_000, None, "fast"))
    output = segmenter.flush("fast")[0]
    assert output.last_speech_end_monotonic is None
    assert output.audio_start_monotonic is None


def test_activation_keeps_full_candidate_and_pre_pad() -> None:
    pre_pad_frames = 7
    activation_frames = 8
    probabilities = [0.1] * pre_pad_frames + [0.9] * activation_frames
    segmenter = VadSegmenter(
        FakePredictor(probabilities),
        16_000,
        512,
        {
            "threshold": 0.5,
            "min_speech_ms": 250,
            "min_silence_ms": 650,
            "max_speech_sec": 10,
            "pre_pad_ms": 200,
            "min_segment_ms": 250,
        },
    )
    frames = [np.full(512, index, dtype=np.float32) for index in range(len(probabilities))]

    segmenter.process(AudioChunk(np.concatenate(frames), 16_000, 100.0, "realtime"))
    segment = segmenter.flush("realtime")[0]

    assert len(segment.samples) == (pre_pad_frames + activation_frames) * 512
    assert [segment.samples[index * 512] for index in range(15)] == list(range(15))
    assert segment.audio_start_monotonic == 100.0
    assert segment.audio_duration_sec == 0.480
    assert segment.speech_frame_sec == 0.256
    assert segment.speech_span_sec == 0.256
    assert abs((segment.last_speech_end_monotonic or 0) - 100.480) < 1e-6


def test_vad_metrics_separate_idle_compute_speech_frames_and_span(monkeypatch) -> None:
    class Clock:
        value = 0.0

        def __call__(self) -> float:
            self.value += 0.001
            return self.value

    monkeypatch.setattr(vad_module.time, "perf_counter", Clock())
    probabilities = [0.1, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1]
    segmenter = VadSegmenter(
        FakePredictor(probabilities),
        16_000,
        512,
        {
            "min_speech_ms": 32,
            "min_silence_ms": 64,
            "pre_pad_ms": 32,
            "min_segment_ms": 32,
        },
    )

    outputs = segmenter.process(
        AudioChunk(np.ones(512 * len(probabilities), dtype=np.float32), 16_000, None, "fast")
    )

    segment = outputs[0]
    assert segment.vad_frame_avg_ms == pytest.approx(1.0)
    assert segment.vad_segment_compute_ms == pytest.approx(6.0)
    assert segment.speech_frame_sec == 0.064
    assert segment.speech_span_sec == 0.096
    assert segment.tail_wait_ms == 64.0


def test_min_segment_uses_accumulated_speech_not_span() -> None:
    probabilities = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
    segmenter = VadSegmenter(
        FakePredictor(probabilities),
        16_000,
        512,
        {
            "min_speech_ms": 32,
            "min_silence_ms": 128,
            "pre_pad_ms": 32,
            "min_segment_ms": 96,
        },
    )

    outputs = segmenter.process(
        AudioChunk(np.ones(512 * len(probabilities), dtype=np.float32), 16_000, None, "fast")
    )

    assert outputs == []
    assert segmenter.discarded_short_segments == 1


def test_input_gap_closes_segment_and_resets_predictor() -> None:
    predictor = FakePredictor([0.9, 0.9])
    segmenter = VadSegmenter(
        predictor,
        16_000,
        512,
        {"min_speech_ms": 32, "min_silence_ms": 64, "pre_pad_ms": 32, "min_segment_ms": 32},
    )
    segmenter.process(AudioChunk(np.ones(512, dtype=np.float32), 16_000, 10.0, "realtime"))

    outputs = segmenter.process(
        AudioChunk(
            np.full(512, 2.0, dtype=np.float32),
            16_000,
            10.064,
            "realtime",
            discontinuity_samples=512,
        )
    )

    assert len(outputs) == 1
    assert outputs[0].end_reason == "input_gap"
    assert predictor.reset_count == 1
    next_segment = segmenter.flush("realtime")[0]
    assert next_segment.samples[0] == 2.0
    assert next_segment.audio_start_monotonic == 10.064


@pytest.mark.parametrize(
    ("speech_frames", "expected_segments", "expected_discarded"),
    [(7, 0, 0), (8, 0, 1), (15, 0, 1), (16, 1, 0)],
)
def test_default_activation_and_final_filter_are_independent(
    speech_frames: int,
    expected_segments: int,
    expected_discarded: int,
) -> None:
    segmenter = VadSegmenter(
        FakePredictor([0.9] * speech_frames),
        16_000,
        512,
        {
            "min_speech_ms": 250,
            "min_silence_ms": 650,
            "max_speech_sec": 15,
            "pre_pad_ms": 200,
            "min_segment_ms": 500,
        },
    )
    segmenter.process(
        AudioChunk(np.ones(512 * speech_frames, dtype=np.float32), 16_000, None, "fast")
    )

    segments = segmenter.flush("fast")

    assert len(segments) == expected_segments
    assert segmenter.discarded_short_segments == expected_discarded


def test_long_speech_is_split_at_max_duration() -> None:
    segmenter = VadSegmenter(
        FakePredictor([0.9] * 6),
        16_000,
        512,
        {
            "min_speech_ms": 32,
            "min_silence_ms": 650,
            "max_speech_sec": 0.096,
            "pre_pad_ms": 32,
            "min_segment_ms": 32,
        },
    )

    segments = segmenter.process(
        AudioChunk(np.ones(512 * 6, dtype=np.float32), 16_000, None, "fast")
    )

    assert len(segments) == 2
    assert [segment.end_reason for segment in segments] == ["max_duration", "max_duration"]
    assert [segment.speech_frame_sec for segment in segments] == [0.096, 0.096]

