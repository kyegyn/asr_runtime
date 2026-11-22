from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class CampPlusEngine:
    name = "CAM++"

    def __init__(self, model_path: Path, config: dict[str, Any]):
        import torch
        from modelscope.models.audio.sv.DTDNN import CAMPPlus

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA 不可用；禁止自动回退 CPU")
        with (model_path / "configuration.json").open("r", encoding="utf-8") as handle:
            official = json.load(handle)["model"]
        model_config = official["model_config"]
        self.feature_dim = int(model_config["fbank_dim"])
        self.embedding_size = int(model_config["emb_size"])
        self.device = torch.device(str(config.get("device", "cuda:0")))
        started = time.perf_counter()
        self.model = CAMPPlus(self.feature_dim, self.embedding_size)
        weight_path = model_path / str(official["pretrained_model"])
        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        torch.cuda.synchronize(self.device.index or 0)
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def _fbank(self, samples: np.ndarray):
        import torchaudio.compliance.kaldi as kaldi

        waveform = self.torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        feature = kaldi.fbank(waveform, num_mel_bins=self.feature_dim)
        return (feature - feature.mean(dim=0, keepdim=True)).unsqueeze(0)

    def infer(self, samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
        if sample_rate != 16_000:
            raise ValueError(f"CAM++ 输入必须为 16kHz，实际 {sample_rate}")
        device_index = self.device.index or 0
        self.torch.cuda.synchronize(device_index)
        total_started = time.perf_counter()

        fbank_started = time.perf_counter()
        features_cpu = self._fbank(samples)
        fbank_ms = (time.perf_counter() - fbank_started) * 1000.0

        h2d_started = time.perf_counter()
        features = features_cpu.to(self.device)
        self.torch.cuda.synchronize(device_index)
        h2d_ms = (time.perf_counter() - h2d_started) * 1000.0

        event_start = self.torch.cuda.Event(enable_timing=True)
        event_end = self.torch.cuda.Event(enable_timing=True)
        forward_started = time.perf_counter()
        event_start.record()
        with self.torch.inference_mode():
            embedding = self.model(features)
        event_end.record()
        event_end.synchronize()
        forward_wall_ms = (time.perf_counter() - forward_started) * 1000.0
        forward_gpu_ms = float(event_start.elapsed_time(event_end))

        d2h_started = time.perf_counter()
        embedding_cpu = embedding.detach().float().cpu()
        self.torch.cuda.synchronize(device_index)
        d2h_ms = (time.perf_counter() - d2h_started) * 1000.0
        vector = embedding_cpu.numpy().reshape(-1).astype(np.float32, copy=False)
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return {
            "embedding": vector,
            "embedding_dim": int(vector.size),
            "embedding_norm": float(np.linalg.norm(vector)),
            "latency_ms": total_ms,
            "fbank_ms": fbank_ms,
            "h2d_ms": h2d_ms,
            "forward_gpu_ms": forward_gpu_ms,
            "forward_wall_ms": forward_wall_ms,
            "d2h_ms": d2h_ms,
        }

