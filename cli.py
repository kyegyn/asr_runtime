from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio.capture import MicrophoneCapture, microphone_devices
from .audio.common import load_audio
from .audio.types import AudioChunk, SpeechSegment
from .audio.vad import SileroVadOnnx, VadSegmenter
from .audio.wav_replay import replay_dataset, replay_wav
from .config import ConfigError, load_config, resolve_paths
from .engines import CampPlusEngine
from .observability.logger import LogManager, setup_logging
from .runtime.aggregator import ResultAggregator
from .runtime.scheduler import ParallelScheduler
from .speaker import SpeakerMatcher, SpeakerRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Silero VAD + SenseVoiceSmall + CAM++ 独立运行时")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="检查模型、依赖、GPU、VAD 和麦克风设备")

    run = sub.add_parser("run", help="运行连续语音识别")
    run.add_argument("--source", choices=["wav", "dataset", "mic"], required=True)
    run.add_argument("--input", type=Path, default=None, help="WAV 文件路径")
    run.add_argument("--limit", type=int, default=None, help="dataset 最大音频数")
    run.add_argument("--realtime", action="store_true", help="WAV/dataset 按音频时间回放")

    enroll = sub.add_parser("enroll", help="用一条 WAV 注册说话人；可重复执行以添加多条声纹")
    enroll.add_argument("--speaker-id", required=True)
    enroll.add_argument("--input", type=Path, required=True)

    sub.add_parser("list-speakers", help="列出本地声纹库")
    return parser


def _key_config(config: dict[str, Any], paths: Any) -> dict[str, Any]:
    return {
        "vad": config["vad"],
        "asr": config["sensevoice"],
        "sid": config["campplus"],
        "workers": config["workers"],
        "speaker_threshold": config["speaker"]["match_threshold"],
        "models": {
            "vad": str(paths.silero_vad_model / "silero_vad.onnx"),
            "asr": str(paths.sensevoice_model / "model.pt"),
            "sid": str(paths.campplus_model / "campplus_cn_common.bin"),
        },
    }


def _environment_check(config: dict[str, Any], paths: Any) -> tuple[dict[str, Any], bool]:
    errors: list[str] = []
    imports: dict[str, str] = {}
    for module in ("torch", "torchaudio", "onnxruntime", "funasr", "modelscope", "soundfile", "sounddevice"):
        try:
            imported = __import__(module)
            imports[module] = str(getattr(imported, "__version__", "installed"))
        except BaseException as exc:
            errors.append(f"import {module}: {type(exc).__name__}: {exc}")
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
        if not torch.cuda.is_available():
            errors.append("PyTorch CUDA 不可用；运行时不会回退 CPU")
    except BaseException as exc:
        errors.append(f"PyTorch GPU 检查失败: {exc}")

    vad_info: dict[str, Any] = {}
    try:
        vad = SileroVadOnnx(paths.silero_vad_model / "silero_vad.onnx")
        probability = vad(np.zeros(512, dtype=np.float32))
        vad_info = {"ok": True, "load_ms": vad.load_ms, "silence_probability": probability}
    except BaseException as exc:
        vad_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"Silero VAD: {vad_info['error']}")

    try:
        devices = microphone_devices()
    except BaseException as exc:
        devices = []
        errors.append(f"麦克风枚举失败: {type(exc).__name__}: {exc}")
    report = {
        "ok": not errors,
        "errors": errors,
        "imports": imports,
        "torch": torch_info,
        "vad": vad_info,
        "microphones": devices,
        "paths": {name: str(value) for name, value in vars(paths).items()},
        "effective_config": _key_config(config, paths),
    }
    return report, not errors


def _source_chunks(args: argparse.Namespace, config: dict[str, Any], paths: Any) -> tuple[Iterable[AudioChunk], Any]:
    sample_rate = int(config["audio"]["sample_rate"])
    frame_samples = int(config["audio"]["frame_samples"])
    if args.source == "wav":
        if args.input is None:
            raise ValueError("--source wav 必须提供 --input")
        input_path = args.input.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"WAV 不存在: {input_path}")
        return replay_wav(input_path, sample_rate, frame_samples, args.realtime), None
    if args.source == "dataset":
        return (
            replay_dataset(
                paths.manifest,
                paths.test_audio_root,
                sample_rate,
                frame_samples,
                int(config["input"].get("wav_gap_ms", 900)),
                args.limit,
                args.realtime,
            ),
            None,
        )
    capture = MicrophoneCapture(
        sample_rate,
        frame_samples,
        int(config["input"].get("microphone_queue_frames", 128)),
        config["input"].get("microphone_device"),
    )
    capture.__enter__()
    return iter(capture), capture


def _submit_segments(
    segments: list[SpeechSegment], aggregator: ResultAggregator, scheduler: ParallelScheduler
) -> None:
    for segment in segments:
        aggregator.register(segment)
        if not scheduler.submit(segment):
            aggregator.reject(segment.segment_id, "scheduler submission queue full")


def _run(args: argparse.Namespace, config: dict[str, Any], paths: Any, log_manager: LogManager) -> int:
    logger = logging.getLogger("runtime")
    startup_started = time.perf_counter()
    logger.info("实际生效配置: %s", json.dumps(_key_config(config, paths), ensure_ascii=False))

    vad_model = SileroVadOnnx(paths.silero_vad_model / "silero_vad.onnx")
    vad_warmup_started = time.perf_counter()
    vad_model(np.zeros(512, dtype=np.float32))
    vad_warmup_ms = (time.perf_counter() - vad_warmup_started) * 1000.0
    vad_model.reset_states()
    segmenter = VadSegmenter(
        vad_model,
        int(config["audio"]["sample_rate"]),
        int(config["audio"]["frame_samples"]),
        config["vad"],
    )
    registry = SpeakerRegistry(paths.speaker_registry_root)
    matcher = SpeakerMatcher(registry, float(config["speaker"]["match_threshold"]))
    aggregator = ResultAggregator(
        matcher,
        float(config["campplus"].get("min_audio_sec", 1.0)),
        bool(config["logging"].get("output_text", True)),
    )
    scheduler = ParallelScheduler(config, paths, log_manager.queue, aggregator.accept)
    import torch

    startup_ms = (time.perf_counter() - startup_started) * 1000.0
    ready = scheduler.startup_info
    logger.info(
        "\n%s",
        "\n".join(
            [
                "[T0 模型初始化]",
                f"  VAD模型          Silero VAD / CPU / load={vad_model.load_ms:.1f}ms / warmup={vad_warmup_ms:.1f}ms",
                f"  ASR模型          SenseVoiceSmall / {ready['asr']['device']} / load={ready['asr']['load_ms']:.1f}ms / warmup={ready['asr']['warmup_ms']:.1f}ms",
                f"  SID模型          CAM++ / {ready['sid']['device']} / load={ready['sid']['load_ms']:.1f}ms / warmup={ready['sid']['warmup_ms']:.1f}ms",
                f"  GPU              {torch.cuda.get_device_name(0)} / initialized={torch.cuda.is_available()}",
                f"  整体启动          {startup_ms:.1f}ms",
            ]
        ),
    )

    source: Any = None
    last_mode = "realtime" if args.source == "mic" or args.realtime else "fast"
    interrupted = False
    try:
        chunks, source = _source_chunks(args, config, paths)
        for chunk in chunks:
            last_mode = chunk.source_mode
            _submit_segments(segmenter.process(chunk), aggregator, scheduler)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("收到 Ctrl+C，停止音频输入并等待在途任务")
    finally:
        if source is not None:
            source.close()
        _submit_segments(
            segmenter.flush(last_mode, "interrupted" if interrupted else "audio_end"),
            aggregator,
            scheduler,
        )
        scheduler.close()
        aggregator.wait_all(timeout=float(config["workers"].get("inference_timeout_sec", 120)) + 5)
        aggregator.log_summary()
        logger.info("模型调度统计: %s", json.dumps(scheduler.stats(), ensure_ascii=False))
        if source is not None and callable(getattr(source, "stats", None)):
            logger.info("麦克风采集统计: %s", json.dumps(source.stats(), ensure_ascii=False))
    return 0


def _enroll(args: argparse.Namespace, config: dict[str, Any], paths: Any) -> int:
    logger = logging.getLogger("enroll")
    input_path = args.input.resolve()
    samples = load_audio(input_path, 16_000)
    min_audio = float(config["campplus"].get("min_audio_sec", 1.0))
    if len(samples) / 16_000 < min_audio:
        raise ValueError(f"注册音频过短，至少需要 {min_audio:.2f}s")
    engine = CampPlusEngine(paths.campplus_model, config["campplus"])
    output = engine.infer(samples, 16_000)
    registry = SpeakerRegistry(paths.speaker_registry_root)
    stored = registry.enroll(args.speaker_id, output["embedding"], str(input_path))
    logger.info(
        "声纹注册成功 speaker_id=%s embedding_dim=%s latency=%.1fms file=%s",
        args.speaker_id,
        output["embedding_dim"],
        output["latency_ms"],
        stored,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    args = _parser().parse_args(argv)
    log_manager: LogManager | None = None
    try:
        config = load_config(args.config)
        paths = resolve_paths(config)
        log_manager = setup_logging(paths.log_root, str(config.get("logging", {}).get("level", "INFO")))
        logger = logging.getLogger("cli")
        logger.info("日志文件: %s", log_manager.path)
        if args.command == "check":
            report, ok = _environment_check(config, paths)
            logger.info("环境检查结果:\n%s", json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if ok else 1
        if args.command == "run":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit 必须大于 0")
            return _run(args, config, paths, log_manager)
        if args.command == "enroll":
            return _enroll(args, config, paths)
        if args.command == "list-speakers":
            speakers = SpeakerRegistry(paths.speaker_registry_root).list_speakers()
            logger.info("已注册说话人: %s", json.dumps(speakers, ensure_ascii=False))
            return 0
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        logging.getLogger("cli").exception("运行失败: %s", exc)
        return 1
    finally:
        if log_manager is not None:
            log_manager.stop()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

