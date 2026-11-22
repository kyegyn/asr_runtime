from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .registry import SpeakerRegistry


@dataclass(frozen=True)
class MatchResult:
    speaker_id: str
    similarity: float | None
    threshold: float
    status: str
    latency_ms: float


class SpeakerMatcher:
    def __init__(self, registry: SpeakerRegistry, threshold: float):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("声纹匹配阈值必须在 [-1, 1]")
        self.registry = registry
        self.threshold = threshold

    def match(self, embedding: np.ndarray | None) -> MatchResult:
        started = time.perf_counter()
        enrolled = self.registry.embeddings()
        if not enrolled:
            return MatchResult("unknown", None, self.threshold, "not_enrolled", (time.perf_counter() - started) * 1000)
        if embedding is None:
            return MatchResult("unknown", None, self.threshold, "unknown", (time.perf_counter() - started) * 1000)
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            return MatchResult("unknown", None, self.threshold, "unknown", (time.perf_counter() - started) * 1000)
        query = query / norm
        best_id = "unknown"
        best_score = -1.0
        for speaker_id, vectors in enrolled.items():
            for vector in vectors:
                score = float(np.dot(query, vector))
                if score > best_score:
                    best_id, best_score = speaker_id, score
        matched = best_score >= self.threshold
        return MatchResult(
            best_id if matched else "unknown",
            best_score,
            self.threshold,
            "matched" if matched else "unknown",
            (time.perf_counter() - started) * 1000,
        )

