from pathlib import Path

import numpy as np

from asr_runtime.speaker import SpeakerMatcher, SpeakerRegistry


def test_registry_supports_multiple_embeddings_and_rejection(tmp_path: Path) -> None:
    registry = SpeakerRegistry(tmp_path)
    registry.enroll("person_01", np.array([1.0, 0.0], dtype=np.float32), "a.wav")
    registry.enroll("person_01", np.array([0.9, 0.1], dtype=np.float32), "b.wav")
    matcher = SpeakerMatcher(registry, 0.8)
    matched = matcher.match(np.array([1.0, 0.0], dtype=np.float32))
    assert matched.status == "matched"
    assert matched.speaker_id == "person_01"
    rejected = matcher.match(np.array([0.0, 1.0], dtype=np.float32))
    assert rejected.status == "unknown"
    assert rejected.speaker_id == "unknown"
    assert registry.list_speakers() == {"person_01": 2}


def test_empty_registry_is_not_enrolled(tmp_path: Path) -> None:
    result = SpeakerMatcher(SpeakerRegistry(tmp_path), 0.7).match(np.ones(2, dtype=np.float32))
    assert result.status == "not_enrolled"
    assert result.similarity is None

