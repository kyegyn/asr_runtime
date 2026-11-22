import logging
from types import SimpleNamespace

import numpy as np

from asr_runtime.audio.capture import MicrophoneCapture


def test_queue_drop_is_reported_and_marks_next_delivered_chunk(caplog) -> None:
    capture = MicrophoneCapture(16_000, 512, queue_frames=1)
    frame = np.ones((512, 1), dtype=np.float32)
    capture._callback(frame, 512, None, None)
    capture._callback(frame * 2, 512, None, None)

    iterator = iter(capture)
    with caplog.at_level(logging.WARNING, logger="audio.capture"):
        first = next(iterator)
    capture._callback(frame * 3, 512, None, None)
    after_gap = next(iterator)
    capture.close()

    assert first.discontinuity_samples == 0
    assert after_gap.samples[0] == 3.0
    assert after_gap.discontinuity_samples == 512
    assert capture.dropped_frames == 1
    assert capture.dropped_samples == 512
    assert "麦克风队列状态" in caplog.text


def test_device_overflow_marks_stream_discontinuity() -> None:
    capture = MicrophoneCapture(16_000, 512, queue_frames=1)
    frame = np.ones((512, 1), dtype=np.float32)

    capture._callback(frame, 512, None, SimpleNamespace(input_overflow=True))
    chunk = next(iter(capture))
    capture.close()

    assert chunk.discontinuity_samples == 512
    assert capture.overflow_events == 1
