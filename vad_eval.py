from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio.common import load_audio
from .audio.types import AudioChunk
from .audio.vad import SileroVadOnnx, VadSegmenter
from .config import load_config, resolve_paths


@dataclass(frozen=True)
class EvaluationCase:
    path: Path
    expected_speech: bool
    duration_sec: float
    source_type: str


def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "speech", "positive"}:
        return True
    if normalized in {"0", "false", "no", "noise", "negative"}:
        return False
    raise ValueError(f"无法解析 expected_speech={value!r}")


def _duration_bucket(duration_sec: float) -> str:
    if duration_sec < 0.25:
        return "<250ms"
    if duration_sec < 0.5:
        return "250-500ms"
    if duration_sec < 1.0:
        return "500-1000ms"
    return ">=1000ms"


def summarize_results(cases: list[EvaluationCase], detected: list[bool]) -> dict[str, Any]:
    if len(cases) != len(detected):
        raise ValueError("cases 与 detected 数量不一致")
    positives = [index for index, case in enumerate(cases) if case.expected_speech]
    negatives = [index for index, case in enumerate(cases) if not case.expected_speech]
    missed = [index for index in positives if not detected[index]]
    false_triggers = [index for index in negatives if detected[index]]
    buckets: dict[str, dict[str, Any]] = {}
    for label in ("<250ms", "250-500ms", "500-1000ms", ">=1000ms"):
        indexes = [
            index
            for index in positives
            if _duration_bucket(cases[index].duration_sec) == label
        ]
        misses = sum(not detected[index] for index in indexes)
        buckets[label] = {
            "samples": len(indexes),
            "misses": misses,
            "miss_rate": misses / len(indexes) if indexes else None,
        }
    populated_positive_buckets = sum(item["samples"] > 0 for item in buckets.values())
    human_positive_buckets = len(
        {
            _duration_bucket(case.duration_sec)
            for case in cases
            if case.expected_speech and case.source_type.lower() == "human"
        }
    )
    human_short_commands = sum(
        case.expected_speech and case.duration_sec < 1.0 and case.source_type.lower() == "human"
        for case in cases
    )
    validation = {
        "human_short_commands": human_short_commands,
        "negative_samples": len(negatives),
        "positive_duration_buckets": populated_positive_buckets,
        "human_duration_buckets": human_positive_buckets,
    }
    validation["supports_threshold_adjustment"] = bool(
        human_short_commands and negatives and human_positive_buckets >= 2
    )
    return {
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "misses": len(missed),
        "false_triggers": len(false_triggers),
        "miss_rate": len(missed) / len(positives) if positives else None,
        "false_trigger_rate": len(false_triggers) / len(negatives) if negatives else None,
        "duration_buckets": buckets,
        "missed_files": [str(cases[index].path) for index in missed],
        "false_trigger_files": [str(cases[index].path) for index in false_triggers],
        "validation": validation,
    }


def _load_cases(manifest: Path, audio_root: Path, assume_positive: bool) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = str(row.get("file", "")).strip()
            if not relative_path:
                raise ValueError("评测清单每行都必须包含 file")
            expected_value = str(row.get("expected_speech", "")).strip()
            if not expected_value and not assume_positive:
                raise ValueError("评测清单必须包含 expected_speech；仅回归测试可使用 --assume-positive")
            path = (audio_root / relative_path).resolve()
            samples = load_audio(path, 16_000)
            duration_value = str(row.get("speech_duration_sec") or row.get("duration_sec") or "").strip()
            duration_sec = float(duration_value) if duration_value else len(samples) / 16_000
            cases.append(
                EvaluationCase(
                    path=path,
                    expected_speech=True if not expected_value else _bool_value(expected_value),
                    duration_sec=duration_sec,
                    source_type=str(row.get("source_type", "unknown")).strip() or "unknown",
                )
            )
    return cases


def _values(value: str | None, default: float) -> list[float]:
    if not value:
        return [default]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估 VAD 漏检率、误触发率与短指令时长分布")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--min-speech-ms", default=None, help="逗号分隔的候选值")
    parser.add_argument("--min-segment-ms", default=None, help="逗号分隔的候选值")
    parser.add_argument("--assume-positive", action="store_true", help="仅用于全语音回归清单")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = resolve_paths(config)
    manifest = (args.manifest or paths.manifest).resolve()
    audio_root = (args.audio_root or paths.test_audio_root).resolve()
    cases = _load_cases(manifest, audio_root, args.assume_positive)
    cached_audio = [load_audio(case.path, 16_000) for case in cases]
    vad_defaults = config["vad"]
    reports = []
    for min_speech_ms in _values(args.min_speech_ms, float(vad_defaults["min_speech_ms"])):
        for min_segment_ms in _values(args.min_segment_ms, float(vad_defaults["min_segment_ms"])):
            predictor = SileroVadOnnx(paths.silero_vad_model / "silero_vad.onnx")
            detected: list[bool] = []
            for samples in cached_audio:
                predictor.reset_states()
                vad_config = dict(vad_defaults)
                vad_config["min_speech_ms"] = min_speech_ms
                vad_config["min_segment_ms"] = min_segment_ms
                segmenter = VadSegmenter(predictor, 16_000, int(config["audio"]["frame_samples"]), vad_config)
                segments = segmenter.process(AudioChunk(samples, 16_000, None, "fast"))
                segments.extend(segmenter.flush("fast"))
                detected.append(bool(segments))
            report = summarize_results(cases, detected)
            report["min_speech_ms"] = min_speech_ms
            report["min_segment_ms"] = min_segment_ms
            reports.append(report)
    print(json.dumps({"manifest": str(manifest), "reports": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
