from pathlib import Path

from asr_runtime.vad_eval import EvaluationCase, summarize_results


def test_summary_reports_duration_misses_and_false_triggers() -> None:
    cases = [
        EvaluationCase(Path("stop.wav"), True, 0.2, "human"),
        EvaluationCase(Path("wait.wav"), True, 0.7, "human"),
        EvaluationCase(Path("noise.wav"), False, 2.0, "environment"),
    ]

    result = summarize_results(cases, [False, True, True])

    assert result["miss_rate"] == 0.5
    assert result["false_trigger_rate"] == 1.0
    assert result["duration_buckets"]["<250ms"]["miss_rate"] == 1.0
    assert result["validation"]["supports_threshold_adjustment"] is True
