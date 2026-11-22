from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from ..audio.common import load_audio
from ..engines import CampPlusEngine, SenseVoiceEngine
from ..observability.logger import configure_process_logging


def _cuda_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in ("cuda", "cudnn", "cublas", "out of memory"))


def _engine(name: str, model_path: Path, config: dict[str, Any]):
    if name == "asr":
        return SenseVoiceEngine(model_path, config)
    if name == "sid":
        return CampPlusEngine(model_path, config)
    raise ValueError(f"未知 Worker: {name}")


def worker_main(
    name: str,
    model_path: str,
    model_config: dict[str, Any],
    warmup_audio: str,
    task_queue: Any,
    result_queue: Any,
    log_queue: Any,
    log_level: str,
) -> None:
    configure_process_logging(log_queue, log_level)
    logger = logging.getLogger(f"worker.{name}")
    try:
        engine = _engine(name, Path(model_path), model_config)
        warmup_samples = load_audio(Path(warmup_audio), 16_000)
        warmup_started = time.perf_counter()
        warmup_output = engine.infer(warmup_samples, 16_000)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        result_queue.put(
            {
                "kind": "ready",
                "ok": True,
                "name": name,
                "model": engine.name,
                "device": str(model_config.get("device", "cuda:0")),
                "load_ms": engine.load_ms,
                "warmup_ms": warmup_ms,
                "warmup_inference_ms": warmup_output.get("latency_ms"),
            }
        )
        logger.info("%s Worker 已就绪，load=%.1fms warmup=%.1fms", name.upper(), engine.load_ms, warmup_ms)
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "ready",
                "ok": False,
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "cuda_error": _cuda_error(exc),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            return
        started_ns = time.perf_counter_ns()
        try:
            output = engine.infer(task["samples"], int(task["sample_rate"]))
            result_queue.put(
                {
                    "kind": "result",
                    "ok": True,
                    "name": name,
                    "segment_id": task["segment_id"],
                    "submitted_ns": task["submitted_ns"],
                    "worker_started_ns": started_ns,
                    "worker_ended_ns": time.perf_counter_ns(),
                    "output": output,
                }
            )
        except BaseException as exc:
            result_queue.put(
                {
                    "kind": "result",
                    "ok": False,
                    "name": name,
                    "segment_id": task["segment_id"],
                    "submitted_ns": task["submitted_ns"],
                    "worker_started_ns": started_ns,
                    "worker_ended_ns": time.perf_counter_ns(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "cuda_error": _cuda_error(exc),
                }
            )


class WorkerClient:
    def __init__(
        self,
        context: mp.context.BaseContext,
        name: str,
        model_path: Path,
        model_config: dict[str, Any],
        warmup_audio: Path,
        queue_size: int,
        timeout_sec: float,
        log_queue: Any,
        log_level: str,
    ):
        self.context = context
        self.name = name
        self.model_path = model_path
        self.model_config = model_config
        self.warmup_audio = warmup_audio
        self.queue_size = queue_size
        self.timeout_sec = timeout_sec
        self.log_queue = log_queue
        self.log_level = log_level
        self.task_queue: Any = None
        self.result_queue: Any = None
        self.process: mp.Process | None = None
        self.ready: dict[str, Any] = {}
        self.generation = 0
        self.restarts = 0
        self._lock = threading.RLock()
        self.start()

    def start(self) -> None:
        with self._lock:
            self.task_queue = self.context.Queue(maxsize=self.queue_size)
            self.result_queue = self.context.Queue(maxsize=self.queue_size)
            self.process = self.context.Process(
                target=worker_main,
                name=f"asr-runtime-{self.name}",
                args=(
                    self.name,
                    str(self.model_path),
                    self.model_config,
                    str(self.warmup_audio),
                    self.task_queue,
                    self.result_queue,
                    self.log_queue,
                    self.log_level,
                ),
            )
            self.process.start()
            try:
                ready = self.result_queue.get(timeout=self.timeout_sec)
            except queue.Empty as exc:
                self._terminate_locked()
                raise TimeoutError(f"{self.name} Worker 启动超时") from exc
            if not ready.get("ok"):
                self._terminate_locked()
                raise RuntimeError(f"{self.name} Worker 加载失败: {ready.get('error')}\n{ready.get('traceback', '')}")
            self.ready = ready
            self.generation += 1

    def submit(self, segment_id: str, samples: Any, sample_rate: int, enqueue_timeout: float) -> int:
        with self._lock:
            if not self.process or not self.process.is_alive():
                self.restart()
            submitted_ns = time.perf_counter_ns()
            self.task_queue.put(
                {
                    "segment_id": segment_id,
                    "samples": samples,
                    "sample_rate": sample_rate,
                    "submitted_ns": submitted_ns,
                },
                timeout=enqueue_timeout,
            )
            return self.generation

    def receive(self, expected_segment_id: str) -> dict[str, Any]:
        result_queue = self.result_queue
        try:
            result = result_queue.get(timeout=self.timeout_sec)
        except queue.Empty:
            self.restart()
            return {
                "ok": False,
                "name": self.name,
                "segment_id": expected_segment_id,
                "error": f"{self.name} inference timeout after {self.timeout_sec}s; worker restarted",
                "timeout": True,
                "cuda_error": False,
            }
        received_ns = time.perf_counter_ns()
        if result.get("segment_id") != expected_segment_id:
            return {
                "ok": False,
                "name": self.name,
                "segment_id": expected_segment_id,
                "error": f"segment_id 不匹配: expected={expected_segment_id}, actual={result.get('segment_id')}",
                "timeout": False,
                "cuda_error": False,
            }
        result["received_ns"] = received_ns
        result["queue_wait_ms"] = (result["worker_started_ns"] - result["submitted_ns"]) / 1e6
        result["result_return_ms"] = (received_ns - result["worker_ended_ns"]) / 1e6
        result["total_ms"] = (received_ns - result["submitted_ns"]) / 1e6
        return result

    def restart(self) -> None:
        with self._lock:
            self._terminate_locked()
            self.restarts += 1
            self.start()

    def _terminate_locked(self) -> None:
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
        for ipc_queue in (self.task_queue, self.result_queue):
            if ipc_queue is not None:
                try:
                    ipc_queue.cancel_join_thread()
                    ipc_queue.close()
                except Exception:
                    pass
        self.process = None
        self.task_queue = None
        self.result_queue = None

    def close(self, shutdown_timeout: float) -> None:
        with self._lock:
            if self.process and self.process.is_alive():
                try:
                    self.task_queue.put(None, timeout=1)
                    self.process.join(timeout=shutdown_timeout)
                except Exception:
                    pass
            self._terminate_locked()

