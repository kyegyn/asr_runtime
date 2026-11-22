from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .common import frame_audio, load_audio
from .types import AudioChunk


def _paced_chunks(
    frames: Iterable[np.ndarray],
    sample_rate: int,
    realtime: bool,
) -> Iterator[AudioChunk]:
    stream_started = time.perf_counter()
    sample_cursor = 0
    for frame in frames:
        if realtime:
            deadline = stream_started + sample_cursor / sample_rate
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            frame_time = deadline
            mode = "realtime"
        else:
            frame_time = None
            mode = "fast"
        yield AudioChunk(frame, sample_rate, frame_time, mode)
        sample_cursor += len(frame)


def replay_wav(
    path: Path,
    sample_rate: int,
    frame_samples: int,
    realtime: bool,
) -> Iterator[AudioChunk]:
    samples = load_audio(path, sample_rate)
    yield from _paced_chunks(frame_audio(samples, frame_samples), sample_rate, realtime)


def _manifest_audio_paths(manifest: Path, root: Path, limit: int | None) -> list[Path]:
    aliases = ("file", "wav", "path", "audio", "audio_path", "wav_path", "filename")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        column = next((fields[name] for name in aliases if name in fields), None)
        if column is None:
            raise ValueError(f"manifest 缺少音频路径列: {reader.fieldnames}")
        paths = [(root / row[column]).resolve() for row in reader]
    selected = paths if limit is None else paths[:limit]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"测试音频不存在: {missing[0]}")
    return selected


def replay_dataset(
    manifest: Path,
    root: Path,
    sample_rate: int,
    frame_samples: int,
    gap_ms: int,
    limit: int | None,
    realtime: bool,
) -> Iterator[AudioChunk]:
    paths = _manifest_audio_paths(manifest, root, limit)
    gap_samples = max(frame_samples, int(sample_rate * gap_ms / 1000))
    gap = np.zeros(gap_samples, dtype=np.float32)
    frames: list[np.ndarray] = []
    for path in paths:
        frames.extend(frame_audio(load_audio(path, sample_rate), frame_samples))
        frames.extend(frame_audio(gap, frame_samples))
    yield from _paced_chunks(frames, sample_rate, realtime)

