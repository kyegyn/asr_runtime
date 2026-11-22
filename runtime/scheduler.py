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


@dataclass(frozen=True)
class SegmentSubmission:
    segment_id: str
    samples: Any
    sample_rate: int


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
        self.logger = logging.getLogger("scheduler")
        worker_config = config["workers"]
        self.shutdown_timeout = float(worker_config.get("shutdown_timeout_sec", 10))
        self.queue_size = int(worker_config.get("queue_size", 8))
        self.submission_queue_size = int(
            worker_config.get("submission_queue_size", self.queue_size * 2)
        )
        self.result_queue_size = int(
            worker_config.get(
                "result_queue_size",
                self.submission_queue_size * 2 + self.queue_size * 4,
            )
        )
        self.context = mp.get_context("spawn")
        common = {
            "context": self.context,
            "warmup_audio": paths.warmup_audio,
            "queue_size": self.queue_size,
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
        self._submissions: queue.Queue[SegmentSubmission | None] = queue.Queue(
            maxsize=self.submission_queue_size
        )
        self._pending = {
            "asr": queue.Queue(maxsize=self.queue_size + 1),
            "sid": queue.Queue(maxsize=self.queue_size + 1),
        }
        self._callback_queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(
            maxsize=self.result_queue_size
        )
        self._stats_lock = threading.Lock()
        self._accepted_submissions = 0
        self._dropped_submissions = 0
        self._branch_submit_errors = 0
        self._max_submission_depth = 0
        self._max_callback_depth = 0
        self._max_pending_depth = {"asr": 0, "sid": 0}
        self._backlog_warned = False
        self._closed = False
        self._receiver_threads = [
            threading.Thread(target=self._receiver, args=("asr", self.asr), name="asr-result-receiver", daemon=True),
            threading.Thread(target=self._receiver, args=("sid", self.sid), name="sid-result-receiver", daemon=True),
        ]
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher,
            name="model-task-dispatcher",
            daemon=True,
        )
        self._callback_thread = threading.Thread(
            target=self._callback_worker,
            name="model-result-callback",
            daemon=True,
        )
        self._callback_thread.start()
        for thread in self._receiver_threads:
            thread.start()
        self._dispatcher_thread.start()

    @property
    def startup_info(self) -> dict[str, dict[str, Any]]:
        return {"asr": self.asr.ready, "sid": self.sid.ready}

    def _queue_result(self, branch: str, result: dict[str, Any]) -> None:
        self._callback_queue.put((branch, result))
        with self._stats_lock:
            self._max_callback_depth = max(self._max_callback_depth, self._callback_queue.qsize())

    def _callback_worker(self) -> None:
        while True:
            item = self._callback_queue.get()
            try:
                if item is None:
                    return
                branch, result = item
                try:
                    self.result_callback(branch, result)
                except BaseException:
                    self.logger.exception("%s 结果回调异常", branch.upper())
            finally:
                self._callback_queue.task_done()

    def _submission_error(self, branch: str, segment_id: str, exc: BaseException) -> None:
        with self._stats_lock:
            self._branch_submit_errors += 1
        self._queue_result(
            branch,
            {
                "ok": False,
                "name": branch,
                "segment_id": segment_id,
                "error": f"submit error: {type(exc).__name__}: {exc}",
                "timeout": isinstance(exc, queue.Full),
                "cuda_error": "cuda" in str(exc).lower(),
            },
        )

    def _dispatcher(self) -> None:
        while True:
            submission = self._submissions.get()
            try:
                if submission is None:
                    return
                if self._submissions.qsize() <= self.submission_queue_size // 2:
                    with self._stats_lock:
                        self._backlog_warned = False
                for branch, worker in (("asr", self.asr), ("sid", self.sid)):
                    try:
                        generation = worker.submit(
                            submission.segment_id,
                            submission.samples,
                            submission.sample_rate,
                        )
                        pending_queue = self._pending[branch]
                        pending_queue.put(
                            PendingBranch(
                                submission.segment_id,
                                submission.samples,
                                submission.sample_rate,
                                generation,
                            )
                        )
                        with self._stats_lock:
                            self._max_pending_depth[branch] = max(
                                self._max_pending_depth[branch], pending_queue.qsize()
                            )
                    except BaseException as exc:
                        self._submission_error(branch, submission.segment_id, exc)
            finally:
                self._submissions.task_done()

    def _receiver(self, branch: str, worker: WorkerClient) -> None:
        pending_queue = self._pending[branch]
        while True:
            pending = pending_queue.get()
            try:
                if pending is None:
                    return
                assert isinstance(pending, PendingBranch)
                if pending.generation != worker.generation:
                    generation = worker.submit(pending.segment_id, pending.samples, pending.sample_rate)
                    pending = PendingBranch(
                        pending.segment_id, pending.samples, pending.sample_rate, generation
                    )
                result = worker.receive(pending.segment_id)
                self._queue_result(branch, result)
            except BaseException as exc:
                self.logger.exception("%s 接收线程异常", branch.upper())
                segment_id = getattr(pending, "segment_id", "unknown")
                self._queue_result(
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

    def submit(self, segment: SpeechSegment) -> bool:
        if self._closed:
            raise RuntimeError("Scheduler 已关闭")
        submission = SegmentSubmission(segment.segment_id, segment.samples, segment.sample_rate)
        try:
            self._submissions.put_nowait(submission)
            depth = self._submissions.qsize()
            warn_backlog = False
            with self._stats_lock:
                self._accepted_submissions += 1
                self._max_submission_depth = max(self._max_submission_depth, depth)
                threshold = max(1, (self.submission_queue_size * 3 + 3) // 4)
                if depth >= threshold and not self._backlog_warned:
                    self._backlog_warned = True
                    warn_backlog = True
            if warn_backlog:
                self.logger.warning(
                    "模型提交队列积压 depth=%s/%s pending_asr=%s pending_sid=%s",
                    depth,
                    self.submission_queue_size,
                    self._pending["asr"].qsize(),
                    self._pending["sid"].qsize(),
                )
            return True
        except queue.Full:
            with self._stats_lock:
                self._dropped_submissions += 1
            self.logger.error(
                "模型提交队列已满，丢弃语音段 segment_id=%s depth=%s/%s "
                "pending_asr=%s pending_sid=%s",
                segment.segment_id,
                self._submissions.qsize(),
                self.submission_queue_size,
                self._pending["asr"].qsize(),
                self._pending["sid"].qsize(),
            )
            return False

    def drain(self) -> None:
        self._submissions.join()
        for pending in self._pending.values():
            pending.join()
        self._callback_queue.join()

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = {
                "submission_queue_depth": self._submissions.qsize(),
                "submission_queue_capacity": self.submission_queue_size,
                "max_submission_queue_depth": self._max_submission_depth,
                "accepted_submissions": self._accepted_submissions,
                "dropped_submissions": self._dropped_submissions,
                "branch_submit_errors": self._branch_submit_errors,
                "callback_queue_depth": self._callback_queue.qsize(),
                "callback_queue_capacity": self.result_queue_size,
                "max_callback_queue_depth": self._max_callback_depth,
                "pending_depth": {name: item.qsize() for name, item in self._pending.items()},
                "max_pending_depth": dict(self._max_pending_depth),
            }
        stats["workers"] = {"asr": self.asr.stats(), "sid": self.sid.stats()}
        return stats

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.drain()
        self._submissions.put(None)
        self._dispatcher_thread.join(timeout=5)
        for pending in self._pending.values():
            pending.put(None)
        for thread in self._receiver_threads:
            thread.join(timeout=5)
        self.asr.close(self.shutdown_timeout)
        self.sid.close(self.shutdown_timeout)
        self._callback_queue.put(None)
        self._callback_thread.join(timeout=5)

