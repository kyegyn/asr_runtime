from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from ..audio.types import SpeechSegment
from ..observability.metrics import RuntimeMetrics
from ..observability.tracing import SegmentTrace
from ..speaker.matcher import MatchResult, SpeakerMatcher


REASON_LABELS = {
    "silence": "静音结束",
    "max_duration": "最大长度",
    "audio_end": "音频结束",
    "interrupted": "程序中断",
}


def _value(result: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not result or not result.get("ok"):
        return default
    return result.get("output", {}).get(key, default)


class ResultAggregator:
    def __init__(
        self,
        matcher: SpeakerMatcher,
        sid_min_audio_sec: float,
        output_text: bool = True,
    ):
        self.matcher = matcher
        self.sid_min_audio_sec = sid_min_audio_sec
        self.output_text = output_text
        self.logger = logging.getLogger("aggregator")
        self.metrics = RuntimeMetrics()
        self._traces: dict[str, SegmentTrace] = {}
        self._condition = threading.Condition(threading.RLock())

    def register(self, segment: SpeechSegment) -> None:
        with self._condition:
            self._traces[segment.segment_id] = SegmentTrace(segment, time.perf_counter())
            self.metrics.total_segments += 1
            self.metrics.vad_tail_ms.append(segment.tail_wait_ms)

    def accept(self, branch: str, result: dict[str, Any]) -> None:
        with self._condition:
            trace = self._traces.get(str(result.get("segment_id")))
            if trace is None:
                self.logger.error("收到未知 segment_id: %s", result.get("segment_id"))
                return
            if branch == "asr":
                trace.asr_result = result
            elif branch == "sid":
                trace.sid_result = result
            else:
                raise ValueError(f"未知分支: {branch}")
            if trace.asr_result is not None and trace.sid_result is not None:
                self._finalize(trace)
                self._traces.pop(trace.segment.segment_id, None)
                self._condition.notify_all()

    def _finalize(self, trace: SegmentTrace) -> None:
        aggregate_started = time.perf_counter()
        asr = trace.asr_result or {}
        sid = trace.sid_result or {}
        for name, result in (("ASR", asr), ("SID", sid)):
            if not result.get("ok"):
                trace.errors.append(f"{name}: {result.get('error', 'unknown error')}")
                self.metrics.cuda_errors += int(bool(result.get("cuda_error")))
                self.metrics.timeouts += int(bool(result.get("timeout")))

        embedding = _value(sid, "embedding")
        if trace.segment.valid_speech_sec < self.sid_min_audio_sec:
            match = MatchResult("unknown", None, self.matcher.threshold, "unknown", 0.0)
        else:
            match = self.matcher.match(embedding)

        trace.status = "SUCCESS" if asr.get("ok") and sid.get("ok") else (
            "TIMEOUT" if asr.get("timeout") or sid.get("timeout") else "ERROR"
        )
        aggregate_ms = (time.perf_counter() - aggregate_started) * 1000.0
        trace.output_monotonic = time.perf_counter()
        total_latency_ms = (
            (trace.output_monotonic - trace.segment.last_speech_end_monotonic) * 1000.0
            if trace.segment.last_speech_end_monotonic is not None else None
        )
        asr_branch_ms = asr.get("total_ms")
        sid_branch_ms = sid.get("total_ms")
        if trace.status == "SUCCESS":
            self.metrics.success += 1
            if asr_branch_ms is not None:
                self.metrics.asr_ms.append(float(asr_branch_ms))
            if sid_branch_ms is not None:
                self.metrics.sid_ms.append(float(sid_branch_ms))
            if total_latency_ms is not None:
                self.metrics.total_ms.append(total_latency_ms)
        else:
            self.metrics.failed += 1

        text = str(_value(asr, "text", ""))
        similarity = "N/A" if match.similarity is None else f"{match.similarity:.4f}"
        total_display = "N/A（快速处理模式）" if total_latency_ms is None else f"{total_latency_ms:.1f}ms"
        block = "\n".join(
            [
                f"━━━━━━ 语音段 #{trace.segment.segment_id} ━━━━━━",
                "",
                "[T1 VAD 语音检测]",
                f"  语音时长        {trace.segment.valid_speech_sec:.2f}s",
                f"  VAD计算         {trace.segment.vad_compute_ms:.1f}ms",
                f"  静音收尾        {trace.segment.tail_wait_ms:.1f}ms",
                f"  分段原因        {REASON_LABELS.get(trace.segment.end_reason, trace.segment.end_reason)}",
                "",
                "[T2 ASR 语音识别]",
                "  模型            SenseVoiceSmall",
                f"  排队等待        {float(asr.get('queue_wait_ms') or 0):.1f}ms",
                f"  推理耗时        {float(_value(asr, 'latency_ms', 0)):.1f}ms",
                f"  识别文本        {json.dumps(text, ensure_ascii=False)}",
                "",
                "[T3 SID 声纹提取]",
                "  模型            CAM++",
                f"  FBank特征       {float(_value(sid, 'fbank_ms', 0)):.1f}ms",
                f"  GPU前向         {float(_value(sid, 'forward_gpu_ms', 0)):.1f}ms",
                f"  声纹提取总耗时  {float(_value(sid, 'latency_ms', 0)):.1f}ms",
                f"  Embedding维度   {int(_value(sid, 'embedding_dim', 0))}",
                "",
                "[T4 声纹匹配]",
                f"  说话人          {match.speaker_id}",
                f"  相似度          {similarity}",
                f"  匹配阈值        {match.threshold:.2f}",
                f"  匹配耗时        {match.latency_ms:.1f}ms",
                f"  匹配状态        {match.status}",
                "",
                "[T5 结果汇合与输出]",
                f"  ASR分支         {float(asr_branch_ms or 0):.1f}ms",
                f"  SID分支         {float(sid_branch_ms or 0):.1f}ms",
                f"  结果汇合        {aggregate_ms:.1f}ms",
                f"  语音结束→输出   {total_display}",
                f"  处理状态        {trace.status}",
                *((f"  错误            {'; '.join(trace.errors)}",) if trace.errors else ()),
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]
        )
        self.logger.info("\n%s", block)
        result_payload = {
            "segment_id": trace.segment.segment_id,
            "text": text,
            "speaker_id": match.speaker_id,
            "speaker_similarity": match.similarity,
            "speaker_status": match.status,
            "status": trace.status,
            "total_latency_ms": total_latency_ms,
        }
        if self.output_text:
            self.logger.info("[RESULT] %s", json.dumps(result_payload, ensure_ascii=False))

    def wait_all(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._condition:
            while self._traces:
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def log_summary(self) -> None:
        self.logger.info("\n%s", self.metrics.summary_block())

