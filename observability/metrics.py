from __future__ import annotations

import math
from dataclasses import dataclass, field


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] + (clean[upper] - clean[lower]) * fraction


@dataclass
class RuntimeMetrics:
    total_segments: int = 0
    success: int = 0
    failed: int = 0
    cuda_errors: int = 0
    timeouts: int = 0
    vad_tail_ms: list[float] = field(default_factory=list)
    asr_ms: list[float] = field(default_factory=list)
    sid_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    def pair(self, values: list[float]) -> str:
        p50, p95 = percentile(values, 50), percentile(values, 95)
        if p50 is None or p95 is None:
            return "N/A"
        return f"{p50:.1f}/{p95:.1f}ms"

    def summary_block(self) -> str:
        return "\n".join(
            [
                "[SUMMARY]",
                f"  总识别段数       {self.total_segments}",
                f"  成功/失败        {self.success}/{self.failed}",
                f"  VAD收尾 P50/P95  {self.pair(self.vad_tail_ms)}",
                f"  ASR P50/P95      {self.pair(self.asr_ms)}",
                f"  SID P50/P95      {self.pair(self.sid_ms)}",
                f"  总延迟 P50/P95   {self.pair(self.total_ms)}",
                f"  GPU异常次数      {self.cuda_errors}",
                f"  超时次数         {self.timeouts}",
            ]
        )

