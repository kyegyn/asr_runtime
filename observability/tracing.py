from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..audio.types import SpeechSegment


@dataclass
class SegmentTrace:
    segment: SpeechSegment
    submitted_monotonic: float
    asr_result: dict[str, Any] | None = None
    sid_result: dict[str, Any] | None = None
    output_monotonic: float | None = None
    status: str = "PENDING"
    errors: list[str] = field(default_factory=list)

