from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np


TAG_RE = re.compile(r"<\|[^|>]+\|>")


def clean_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = rich_transcription_postprocess(text)
    except (ImportError, AttributeError):
        pass
    return TAG_RE.sub("", text).strip()


class SenseVoiceEngine:
    name = "SenseVoiceSmall"

    def __init__(self, model_path: Path, config: dict[str, Any]):
        import torch
        from funasr import AutoModel

        self.torch = torch
        self.config = config
        self.device = str(config.get("device", "cuda:0"))
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA 不可用；禁止自动回退 CPU")
        started = time.perf_counter()
        self.model = AutoModel(
            model=str(model_path),
            device=self.device,
            disable_update=True,
            disable_pbar=bool(config.get("disable_pbar", True)),
        )
        torch.cuda.synchronize(0)
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def infer(self, samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
        if sample_rate != 16_000:
            raise ValueError(f"SenseVoice 输入必须为 16kHz，实际 {sample_rate}")
        self.torch.cuda.synchronize(0)
        started = time.perf_counter()
        result = self.model.generate(
            input=np.ascontiguousarray(samples, dtype=np.float32),
            cache={},
            language=str(self.config.get("language", "zh")),
            use_itn=bool(self.config.get("use_itn", True)),
            batch_size_s=int(self.config.get("batch_size_s", 60)),
        )
        self.torch.cuda.synchronize(0)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw = str(result[0].get("text", "")) if result else ""
        return {"text": clean_text(raw), "raw_model_text": raw, "latency_ms": latency_ms}

