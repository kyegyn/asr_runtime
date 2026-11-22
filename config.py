from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
EXAMPLE_CONFIG = PROJECT_ROOT / "config.example.yaml"


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
    config_path = (path or (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else EXAMPLE_CONFIG)).resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"配置文件顶层必须是对象: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def _config_root(config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("_config_path", EXAMPLE_CONFIG)))
    return configured.resolve().parent


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


def _configured_path(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def resolve_paths(config: dict[str, Any]) -> RuntimePaths:
    values = config.get("paths", {})
    config_root = _config_root(config)
    models_root = _configured_path(str(values.get("models_root", "models")), config_root)
    test_audio_root = _configured_path(str(values.get("test_audio_root", "test_audio")), config_root)
    warmup_audio = _configured_path(
        str(config.get("workers", {}).get("warmup_audio", "test_audio/command/001_command.wav")),
        config_root,
    )
    if not warmup_audio.is_file():
        warmup_audio = (test_audio_root / "command" / "001_command.wav").resolve()
    return RuntimePaths(
        project_root=PROJECT_ROOT,
        models_root=models_root,
        test_audio_root=test_audio_root,
        log_root=_configured_path(str(values.get("log_root", "log")), config_root),
        speaker_registry_root=_configured_path(
            str(values.get("speaker_registry_root", "data/speakers")), config_root
        ),
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

