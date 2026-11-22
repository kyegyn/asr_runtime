from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    start_time_monotonic: float | None
    source_mode: Literal["realtime", "fast"]


@dataclass(frozen=True)
class SpeechSegment:
    segment_id: str
    samples: np.ndarray
    sample_rate: int
    valid_speech_sec: float
    vad_compute_ms: float
    tail_wait_ms: float
    end_reason: str
    last_speech_end_monotonic: float | None
    vad_confirmed_monotonic: float
    source_mode: Literal["realtime", "fast"]

