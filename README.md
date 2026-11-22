# ASR Runtime

这是从已验证 V2 方案独立提取的 Windows 语音运行时：Silero VAD 在 CPU 上进行连续检测和自动分段，SenseVoiceSmall 与 CAM++ 在两个长期驻留的 `spawn` Worker 中并行使用 GPU 0。项目不包含 Agent、TTS、UDP 或流式 ASR。

## 环境与 uv

项目使用自己的 Python 3.12 虚拟环境和 `uv.lock`，不会修改 `D:\Work\asr` 的环境。模型和测试音频只从同级目录读取，不会下载或覆盖。

```powershell
cd D:\Work\asr_runtime
$env:UV_CACHE_DIR = "D:\Work\asr_runtime\.uv-cache"
uv sync --extra dev
```

从 `D:\Work` 运行目标 CLI：

```powershell
cd D:\Work
.\asr_runtime\.venv\Scripts\Activate.ps1
python -m asr_runtime.cli check
```

也可以不激活环境：

```powershell
uv run --project .\asr_runtime python -m asr_runtime.cli check
```

## 运行

WAV 按真实音频时间回放：

```powershell
python -m asr_runtime.cli run --source wav --input ".\test_audio\command\001_command.wav" --realtime
```

测试集连续回放：

```powershell
python -m asr_runtime.cli run --source dataset --limit 5 --realtime
```

快速离线功能回归（不会伪装成实时语音结束延迟）：

```powershell
python -m asr_runtime.cli run --source dataset --limit 5
```

Windows 麦克风：

```powershell
python -m asr_runtime.cli run --source mic
```

音频采集回调只把 float32 帧写入有界队列；VAD 和模型推理不会在回调线程运行。按 `Ctrl+C` 后程序停止采集、等待在途任务并输出 SUMMARY。

## 声纹注册

一名说话人可重复注册多条 WAV：

```powershell
python -m asr_runtime.cli enroll --speaker-id person_01 --input ".\test_audio\command\001_command.wav"
python -m asr_runtime.cli enroll --speaker-id person_01 --input ".\test_audio\command\002_command.wav"
python -m asr_runtime.cli list-speakers
```

声纹保存在 `data/speakers/`。余弦匹配阈值位于 `config.yaml`，默认 `0.70` 只是配置起点，必须用真人注册、同人复测和陌生人数据校准。本项目自带的单音色 TTS 数据不能证明真人声纹准确率。

## VAD 与时钟语义

- 输入统一为 16kHz、单声道、float32、每帧 512 samples。
- 默认静音收尾为 650ms，避免自然停顿被过度切分；阈值、最长语音、前沿补偿等均在 `config.yaml`。
- 麦克风和实时 WAV 使用采样位置对应的单调时钟计算“最后有效语音→结果输出”。
- 快速离线模式没有真实采集时钟，T5 明确显示 `N/A（快速处理模式）`，只报告实际推理与调度耗时。

## 日志

每次命令都会在 `log/` 新建 `asr_YYYYMMDD_HHMMSS_PID.log`。主进程和 Worker 统一通过 `QueueHandler + QueueListener` 写 UTF-8 文件并输出到 PowerShell。

- T0：VAD/ASR/SID 加载、预热、设备和整体启动。
- T1：有效语音、VAD 计算、静音收尾和分段原因。
- T2：ASR 排队、推理和文本。
- T3：CAM++ FBank、CUDA Event 前向、总耗时和维度。
- T4：声纹身份、相似度、阈值、状态和耗时。
- T5：两个分支、汇合、真实语音结束延迟和状态。

## 测试

```powershell
cd D:\Work
uv run --project .\asr_runtime python -m pytest .\asr_runtime\tests
uv run --project .\asr_runtime python -m compileall -q .\asr_runtime
```

历史 V2 的 102.56ms ASR P95、46.53ms SID P95 和 103.27ms 汇合 P95 仅是原 benchmark 参考值，新运行时必须以自身日志实测为准。

