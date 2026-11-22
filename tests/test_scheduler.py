import threading
import time
from pathlib import Path

import numpy as np

from asr_runtime.audio.capture import MicrophoneCapture
from asr_runtime.audio.types import SpeechSegment
from asr_runtime.runtime import scheduler as scheduler_module


def _segment(segment_id: str) -> SpeechSegment:
    return SpeechSegment(
        segment_id=segment_id,
        samples=np.ones(512, dtype=np.float32),
        sample_rate=16_000,
        audio_start_monotonic=None,
        audio_duration_sec=0.032,
        speech_frame_sec=0.032,
        speech_span_sec=0.032,
        vad_frame_avg_ms=0.1,
        vad_segment_compute_ms=0.1,
        tail_wait_ms=0.0,
        end_reason="audio_end",
        last_speech_end_monotonic=None,
        vad_confirmed_monotonic=0.0,
        source_mode="fast",
    )


def test_submit_never_waits_for_worker_or_full_queue(monkeypatch) -> None:
    release_worker = threading.Event()
    worker_entered = threading.Event()
    submit_threads: list[str] = []

    class FakeWorkerClient:
        def __init__(self, name: str, **_kwargs):
            self.name = name
            self.ready = {"name": name}
            self.generation = 1

        def submit(self, segment_id: str, samples, sample_rate: int) -> int:
            submit_threads.append(threading.current_thread().name)
            worker_entered.set()
            release_worker.wait(timeout=2)
            return self.generation

        def receive(self, expected_segment_id: str):
            return {"ok": True, "name": self.name, "segment_id": expected_segment_id, "output": {}}

        def close(self, _shutdown_timeout: float) -> None:
            pass

        def stats(self):
            return {"name": self.name, "generation": self.generation, "restarts": 0, "alive": True}

    monkeypatch.setattr(scheduler_module, "WorkerClient", FakeWorkerClient)
    results: list[tuple[str, dict]] = []
    config = {
        "workers": {
            "queue_size": 1,
            "submission_queue_size": 1,
            "inference_timeout_sec": 1,
            "shutdown_timeout_sec": 1,
        },
        "sensevoice": {},
        "campplus": {},
        "logging": {"level": "INFO"},
    }
    paths = type(
        "Paths",
        (),
        {
            "warmup_audio": Path("warmup.wav"),
            "sensevoice_model": Path("sensevoice"),
            "campplus_model": Path("campplus"),
        },
    )()
    scheduler = scheduler_module.ParallelScheduler(config, paths, None, lambda branch, result: results.append((branch, result)))

    started = time.perf_counter()
    assert scheduler.submit(_segment("0001")) is True
    assert time.perf_counter() - started < 0.05
    assert worker_entered.wait(timeout=1)
    assert scheduler.submit(_segment("0002")) is True
    started = time.perf_counter()
    assert scheduler.submit(_segment("0003")) is False
    assert time.perf_counter() - started < 0.05

    release_worker.set()
    scheduler.close()

    assert submit_threads
    assert all(name != threading.current_thread().name for name in submit_threads)
    assert not [result for _branch, result in results if result["segment_id"] == "0003"]
    assert scheduler.stats()["dropped_submissions"] == 1


def test_busy_model_dispatch_does_not_starve_capture_consumer(monkeypatch, caplog) -> None:
    release_worker = threading.Event()
    worker_entered = threading.Event()

    class BlockingWorkerClient:
        def __init__(self, name: str, **_kwargs):
            self.name = name
            self.ready = {"name": name}
            self.generation = 1

        def submit(self, segment_id: str, samples, sample_rate: int) -> int:
            worker_entered.set()
            release_worker.wait(timeout=2)
            return self.generation

        def receive(self, expected_segment_id: str):
            return {"ok": True, "name": self.name, "segment_id": expected_segment_id, "output": {}}

        def close(self, _shutdown_timeout: float) -> None:
            pass

        def stats(self):
            return {"name": self.name, "generation": self.generation, "restarts": 0, "alive": True}

    monkeypatch.setattr(scheduler_module, "WorkerClient", BlockingWorkerClient)
    caplog.set_level("CRITICAL")
    config = {
        "workers": {
            "queue_size": 1,
            "submission_queue_size": 1,
            "result_queue_size": 8,
            "inference_timeout_sec": 1,
            "shutdown_timeout_sec": 1,
        },
        "sensevoice": {},
        "campplus": {},
        "logging": {"level": "INFO"},
    }
    paths = type(
        "Paths",
        (),
        {
            "warmup_audio": Path("warmup.wav"),
            "sensevoice_model": Path("sensevoice"),
            "campplus_model": Path("campplus"),
        },
    )()
    scheduler = scheduler_module.ParallelScheduler(config, paths, None, lambda *_args: None)
    assert scheduler.submit(_segment("initial"))
    assert worker_entered.wait(timeout=1)
    assert scheduler.submit(_segment("queued"))

    capture = MicrophoneCapture(16_000, 512, queue_frames=2)
    capture_iter = iter(capture)
    frame = np.ones((512, 1), dtype=np.float32)
    started = time.perf_counter()
    for index in range(100):
        assert scheduler.submit(_segment(f"overload-{index}")) is False
        capture._callback(frame, 512, None, None)
        next(capture_iter)
    elapsed = time.perf_counter() - started

    capture.close()
    release_worker.set()
    scheduler.close()

    assert elapsed < 1.0
    assert capture.stats()["dropped_frames"] == 0
    assert scheduler.stats()["dropped_submissions"] == 100
