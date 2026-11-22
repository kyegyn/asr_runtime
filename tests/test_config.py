from pathlib import Path

from asr_runtime import config as config_module


def test_load_config_falls_back_to_example(monkeypatch, tmp_path: Path) -> None:
    local_config = tmp_path / "config.yaml"
    example_config = tmp_path / "config.example.yaml"
    example_config.write_text("logging:\n  level: INFO\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", local_config)
    monkeypatch.setattr(config_module, "EXAMPLE_CONFIG", example_config)

    loaded = config_module.load_config()

    assert loaded["logging"]["level"] == "INFO"
    assert loaded["_config_path"] == str(example_config.resolve())


def test_paths_are_relative_to_selected_config(tmp_path: Path) -> None:
    config_root = tmp_path / "settings"
    models_root = config_root / "assets" / "models"
    test_audio_root = config_root / "assets" / "test_audio"
    required_files = {
        "SenseVoiceSmall": ("model.pt", "config.yaml", "am.mvn"),
        "campplus": ("campplus_cn_common.bin", "configuration.json"),
        "silero-vad": ("silero_vad.onnx",),
    }
    for model_name, filenames in required_files.items():
        model_dir = models_root / model_name
        model_dir.mkdir(parents=True)
        for filename in filenames:
            (model_dir / filename).touch()
    warmup_audio = test_audio_root / "command" / "001_command.wav"
    warmup_audio.parent.mkdir(parents=True)
    warmup_audio.touch()

    config = {
        "_config_path": str(config_root / "runtime.yaml"),
        "paths": {
            "models_root": "assets/models",
            "test_audio_root": "assets/test_audio",
            "log_root": "var/log",
            "speaker_registry_root": "var/speakers",
            "sensevoice_candidates": ["SenseVoiceSmall"],
            "campplus_candidates": ["campplus"],
            "silero_vad_candidates": ["silero-vad"],
        },
        "workers": {"warmup_audio": "assets/test_audio/command/001_command.wav"},
    }

    paths = config_module.resolve_paths(config)

    assert paths.models_root == models_root.resolve()
    assert paths.test_audio_root == test_audio_root.resolve()
    assert paths.log_root == (config_root / "var" / "log").resolve()
    assert paths.speaker_registry_root == (config_root / "var" / "speakers").resolve()
    assert paths.warmup_audio == warmup_audio.resolve()
