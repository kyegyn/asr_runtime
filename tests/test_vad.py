import numpy as np

from asr_runtime.audio.types import AudioChunk
from asr_runtime.audio.vad import VadSegmenter


class FakePredictor:
    def __init__(self, probabilities: list[float]):
        self.values = iter(probabilities)

    def __call__(self, _frame: np.ndarray) -> float:
        return next(self.values)


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
    assert segment.valid_speech_sec == 0.064
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

