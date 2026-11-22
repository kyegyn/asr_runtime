from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        moment = datetime.fromtimestamp(record.created)
        return moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


@dataclass
class LogManager:
    queue: Any
    listener: logging.handlers.QueueListener
    path: Path

    def stop(self) -> None:
        self.listener.stop()


def setup_logging(log_root: Path, level_name: str = "INFO") -> LogManager:
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_root / f"asr_{timestamp}_{os.getpid()}.log"
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = MillisecondFormatter("[%(asctime)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    context = mp.get_context("spawn")
    log_queue = context.Queue()
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, console_handler, respect_handler_level=True
    )
    listener.start()
    configure_process_logging(log_queue, level_name)
    return LogManager(log_queue, listener, log_path)


def configure_process_logging(log_queue: Any, level_name: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logging.getLogger("asyncio").setLevel(logging.WARNING)

