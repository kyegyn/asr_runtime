from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: Path, target_sample_rate: int = 16_000) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32)
    if sample_rate != target_sample_rate:
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(mono, dtype=np.float32)


def frame_audio(samples: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset:offset + frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)))
        frames.append(np.ascontiguousarray(frame, dtype=np.float32))
    return frames

