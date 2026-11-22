from asr_runtime.observability.metrics import RuntimeMetrics, percentile


def test_percentile_and_failed_counts_are_separate() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    metrics = RuntimeMetrics(total_segments=2, success=1, failed=1, asr_ms=[10.0])
    text = metrics.summary_block()
    assert "成功/失败        1/1" in text
    assert "10.0/10.0ms" in text

