from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


SPEAKER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class SpeakerRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"version": 1, "speakers": {}}
        with self.index_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("version") != 1 or not isinstance(data.get("speakers"), dict):
            raise ValueError(f"无效的声纹库索引: {self.index_path}")
        return data

    def _save_index(self) -> None:
        temp = self.index_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.index_path)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("声纹 Embedding 范数为 0")
        return vector / norm

    def enroll(self, speaker_id: str, embedding: np.ndarray, source: str | None = None) -> Path:
        if not SPEAKER_ID_RE.fullmatch(speaker_id):
            raise ValueError("speaker_id 只能包含字母、数字、点、下划线和连字符，长度 1–64")
        vector = self._normalize(embedding)
        filename = f"{speaker_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.npy"
        path = (self.root / filename).resolve()
        path.relative_to(self.root)
        np.save(path, vector, allow_pickle=False)
        record = self._index["speakers"].setdefault(speaker_id, {"embeddings": []})
        record["embeddings"].append(
            {"file": filename, "source": source, "created_at_epoch": time.time()}
        )
        self._save_index()
        return path

    def embeddings(self) -> dict[str, list[np.ndarray]]:
        output: dict[str, list[np.ndarray]] = {}
        for speaker_id, record in self._index["speakers"].items():
            vectors: list[np.ndarray] = []
            for item in record.get("embeddings", []):
                path = (self.root / item["file"]).resolve()
                try:
                    path.relative_to(self.root)
                except ValueError:
                    continue
                if path.is_file():
                    vectors.append(self._normalize(np.load(path, allow_pickle=False)))
            if vectors:
                output[speaker_id] = vectors
        return output

    def list_speakers(self) -> dict[str, int]:
        return {speaker_id: len(vectors) for speaker_id, vectors in self.embeddings().items()}

