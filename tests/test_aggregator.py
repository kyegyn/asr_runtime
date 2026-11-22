import numpy as np

from asr_runtime.audio.types import SpeechSegment
from asr_runtime.runtime.aggregator import ResultAggregator


class FakeMatcher:
    threshold = 0.7

    def __init__(self) -> None:
        self.calls = 0

    def match(self, _embedding):
        self.calls += 1
        raise AssertionError("实际语音帧不足时不应执行声纹匹配")


def test_sid_minimum_uses_accumulated_speech_frames() -> None:
    matcher = FakeMatcher()
    aggregator = ResultAggregator(matcher, sid_min_audio_sec=1.0, output_text=False)  # type: ignore[arg-type]
    segment = SpeechSegment(
        segment_id="0001",
        samples=np.ones(32_000, dtype=np.float32),
        sample_rate=16_000,
        audio_start_monotonic=None,
        audio_duration_sec=2.0,
        speech_frame_sec=0.5,
        speech_span_sec=1.8,
        vad_frame_avg_ms=0.1,
        vad_segment_compute_ms=4.0,
        tail_wait_ms=200.0,
        end_reason="silence",
        last_speech_end_monotonic=None,
        vad_confirmed_monotonic=0.0,
        source_mode="fast",
    )
    aggregator.register(segment)

    aggregator.accept(
        "asr",
        {"ok": True, "segment_id": "0001", "output": {"text": "停"}},
    )
    aggregator.accept(
        "sid",
        {"ok": True, "segment_id": "0001", "output": {"embedding": np.ones(2)}},
    )

    assert matcher.calls == 0
    assert aggregator.wait_all(timeout=0.1)
