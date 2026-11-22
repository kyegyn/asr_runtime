from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from ..audio.types import SpeechSegment
from ..config import RuntimePaths
from .workers import WorkerClient


@dataclass(frozen=True)
class PendingBranch:
    segment_id: str
    samples: Any
    sample_rate: int
    generation: int


class ParallelScheduler:
    def __init__(
        self,
        config: dict[str, Any],
        paths: RuntimePaths,
        log_queue: Any,
        result_callback: Callable[[str, dict[str, Any]], None],
    ):
        self.config = config
        self.paths = paths
        self.result_callback = result_callback
        worker_config = config["workers"]
        self.enqueue_timeout = float(worker_config.get("enqueue_timeout_sec", 2))
        self.shutdown_timeout = float(worker_config.get("shutdown_timeout_sec", 10))
        self.context = mp.get_context("spawn")
        common = {
            "context": self.context,
            "warmup_audio": paths.warmup_audio,
            "queue_size": int(worker_config.get("queue_size", 8)),
            "timeout_sec": float(worker_config.get("inference_timeout_sec", 120)),
            "log_queue": log_queue,
            "log_level": str(config.get("logging", {}).get("level", "INFO")),
        }
        self.asr = WorkerClient(
            name="asr", model_path=paths.sensevoice_model, model_config=config["sensevoice"], **common
        )
        self.sid = WorkerClient(
            name="sid", model_path=paths.campplus_model, model_config=config["campplus"], **common
        )
        self._pending = {
            "asr": queue.Queue(maxsize=int(worker_config.get("queue_size", 8)) * 2),
            "sid": queue.Queue(maxsize=int(worker_config.get("queue_size", 8)) * 2),
        }
        self._closed = False
        self._threads = [
            threading.Thread(target=self._receiver, args=("asr", self.asr), name="asr-result-receiver", daemon=True),
            threading.Thread(target=self._receiver, args=("sid", self.sid), name="sid-result-receiver", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    @property
    def startup_info(self) -> dict[str, dict[str, Any]]:
        return {"asr": self.asr.ready, "sid": self.sid.ready}

    def _receiver(self, branch: str, worker: WorkerClient) -> None:
        pending_queue = self._pending[branch]
        while True:
            pending = pending_queue.get()
            try:
                if pending is None:
                    return
                assert isinstance(pending, PendingBranch)
                if pending.generation != worker.generation:
                    generation = worker.submit(
                        pending.segment_id, pending.samples, pending.sample_rate, self.enqueue_timeout
                    )
                    pending = PendingBranch(
                        pending.segment_id, pending.samples, pending.sample_rate, generation
                    )
                result = worker.receive(pending.segment_id)
                self.result_callback(branch, result)
            except BaseException as exc:
                logging.getLogger("scheduler").exception("%s 接收线程异常", branch.upper())
                segment_id = getattr(pending, "segment_id", "unknown")
                self.result_callback(
                    branch,
                    {
                        "ok": False,
                        "name": branch,
                        "segment_id": segment_id,
                        "error": f"receiver error: {type(exc).__name__}: {exc}",
                        "timeout": False,
                        "cuda_error": "cuda" in str(exc).lower(),
                    },
                )
            finally:
                pending_queue.task_done()

    def submit(self, segment: SpeechSegment) -> None:
        if self._closed:
            raise RuntimeError("Scheduler 已关闭")
        for branch, worker in (("asr", self.asr), ("sid", self.sid)):
            try:
                generation = worker.submit(
                    segment.segment_id, segment.samples, segment.sample_rate, self.enqueue_timeout
                )
                self._pending[branch].put(
                    PendingBranch(segment.segment_id, segment.samples, segment.sample_rate, generation),
                    timeout=self.enqueue_timeout,
                )
            except BaseException as exc:
                self.result_callback(
                    branch,
                    {
                        "ok": False,
                        "name": branch,
                        "segment_id": segment.segment_id,
                        "error": f"submit error: {type(exc).__name__}: {exc}",
                        "timeout": isinstance(exc, queue.Full),
                        "cuda_error": "cuda" in str(exc).lower(),
                    },
                )

    def drain(self) -> None:
        for pending in self._pending.values():
            pending.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.drain()
        for pending in self._pending.values():
            pending.put(None)
        for thread in self._threads:
            thread.join(timeout=5)
        self.asr.close(self.shutdown_timeout)
        self.sid.close(self.shutdown_timeout)

