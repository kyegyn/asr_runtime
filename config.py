from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    models_root: Path
    test_audio_root: Path
    log_root: Path
    speaker_registry_root: Path
    sensevoice_model: Path
    campplus_model: Path
    silero_vad_model: Path
    manifest: Path
    warmup_audio: Path


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or PROJECT_ROOT / "config.yaml").resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"配置文件顶层必须是对象: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def _resolve_existing(configured: str, expected_name: str) -> Path:
    value = Path(configured)
    candidates = [
        value if value.is_absolute() else PROJECT_ROOT / value,
        PROJECT_ROOT.parent / expected_name,
        PROJECT_ROOT.parent.parent / expected_name,
    ]
    for candidate in candidates:
        if candidate.resolve().exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _model(root: Path, names: list[str], required: list[str], label: str) -> Path:
    checked: list[Path] = []
    for name in names:
        candidate = (root / name).resolve()
        checked.append(candidate)
        if candidate.is_dir() and all((candidate / item).is_file() for item in required):
            return candidate
    if root.is_dir():
        for candidate in root.iterdir():
            if candidate.is_dir() and all((candidate / item).is_file() for item in required):
                return candidate.resolve()
    raise ConfigError(f"找不到 {label}；检查过: {', '.join(map(str, checked))}")


def _project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def resolve_paths(config: dict[str, Any]) -> RuntimePaths:
    values = config.get("paths", {})
    models_root = _resolve_existing(str(values.get("models_root", "../models")), "models")
    test_audio_root = _resolve_existing(str(values.get("test_audio_root", "../test_audio")), "test_audio")
    warmup_value = Path(str(config.get("workers", {}).get("warmup_audio", "../test_audio/command/001_command.wav")))
    warmup_audio = (
        warmup_value.resolve()
        if warmup_value.is_absolute()
        else (PROJECT_ROOT / warmup_value).resolve()
    )
    if not warmup_audio.is_file():
        warmup_audio = (test_audio_root / "command" / "001_command.wav").resolve()
    return RuntimePaths(
        project_root=PROJECT_ROOT,
        models_root=models_root,
        test_audio_root=test_audio_root,
        log_root=_project_path(str(values.get("log_root", "log"))),
        speaker_registry_root=_project_path(str(values.get("speaker_registry_root", "data/speakers"))),
        sensevoice_model=_model(
            models_root,
            list(values.get("sensevoice_candidates", ["SenseVoiceSmall"])),
            ["model.pt", "config.yaml", "am.mvn"],
            "SenseVoiceSmall",
        ),
        campplus_model=_model(
            models_root,
            list(values.get("campplus_candidates", ["campplus"])),
            ["campplus_cn_common.bin", "configuration.json"],
            "CAM++",
        ),
        silero_vad_model=_model(
            models_root,
            list(values.get("silero_vad_candidates", ["silero-vad"])),
            ["silero_vad.onnx"],
            "Silero VAD",
        ),
        manifest=(test_audio_root / "manifest.csv").resolve(),
        warmup_audio=warmup_audio,
    )

