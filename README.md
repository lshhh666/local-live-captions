# Local Live Captions（本地实时字幕）

Windows 本地实时双语字幕工具。它从系统正在播放的声音中识别英语或俄语，显示原文，
再使用本地 Qwen 模型翻译成简体中文。识别和翻译均在本机完成。

> 当前为 `v0.1` 测试版。已在 Windows 11、RTX 4060 8 GB 上完成真实录屏测试；
> 其他硬件、音频设备和语言场景仍需要更多用户验证。

## 功能

- 捕获 Windows WASAPI 回环声音，不需要虚拟声卡。
- 英语、俄语或自动语言识别。
- 英文/俄文与中文双语置顶悬浮字幕。
- 根据未说完的句子实时修订字幕，`~` 是临时版本，`✓` 是基本确定的版本。
- 本地 Whisper 语音识别和本地 Qwen 翻译，不调用云端字幕接口。
- 默认不保存音频、字幕或个人设置。
- 固定大小的音频缓冲区和翻译队列，长时间运行不会无限占用内存。
- NVIDIA GPU 加速，并提供速度较慢的 CPU 兼容模式。

## 隐私设计

- 音频只存在于固定 30 秒的内存环形缓冲区。
- 翻译上下文只保留最近 5 个已确认句子，并在内存中自动淘汰。
- 本地翻译服务仅监听 `127.0.0.1`。
- 停止或退出时清空本次会话状态。
- 本项目没有自动保存字幕的功能。

## 环境要求

- Windows 11 64 位（Windows 10 可能可用，但尚未完整验证）。
- Python 3.13；发布开发环境使用 Python 3.13.13。
- 建议 NVIDIA 显卡至少 6 GB 显存。
- CPU 模式可以启动，但实时字幕延迟可能明显增加。
- 模型和运行库合计需要约 3 GB 下载空间，完整开发环境需要更多空间。

## 从源码安装

以下命令都在仓库根目录的 PowerShell 中运行。

### 1. 创建 Python 环境

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[gpu-windows]"
```

如果没有 NVIDIA 显卡，可以只安装基础依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. 下载语音识别模型

```powershell
.\scripts\resume-large-v3-turbo.ps1
```

脚本从
[mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)
下载所需文件，支持重新运行，并在启用模型前校验 SHA-256。

### 3. 下载中文翻译模型

```powershell
.\scripts\download-qwen3.ps1
```

脚本下载
[ggml-org/Qwen3-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF)
的 `Qwen3-1.7B-Q4_K_M.gguf`，并放到：

```text
models/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf
```

### 4. 下载 llama.cpp

```powershell
.\scripts\download-llama-runtime.ps1
```

脚本下载项目已经验证的 llama.cpp `b10595` Windows Vulkan 版本，并只提取运行字幕
所需的文件。来源是 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases/tag/b10595)。

模型和运行库都被 `.gitignore` 排除，不会提交到 Git。

## 使用

双击仓库根目录的 `启动字幕工具.cmd`，选择视频语言，然后点击“开始字幕”。
开始播放视频或直播后，字幕窗口会显示在屏幕底部。

也可以使用 PowerShell：

```powershell
.\scripts\test-audio.ps1
.\scripts\run-caption.ps1 -Language en
```

俄语：

```powershell
.\scripts\run-caption.ps1 -Language ru
```

自动识别语言：

```powershell
.\scripts\run-caption.ps1 -Language auto
```

CPU 兼容模式：

```powershell
.\scripts\run-caption.ps1 -Language en -Cpu
```

只显示原文、不加载中文模型：

```powershell
.\scripts\run-caption.ps1 -Language en -NoTranslation
```

点击控制面板的“停止并清空”、字幕窗口右上角的 `×`，或者在控制台按
`Ctrl+C` / `Enter` 均可停止程序。

## 测试

```powershell
$env:PYTHONPATH = "$PWD\py"
.\.venv\Scripts\python.exe -m unittest discover -s tests_py -v
```

当前测试覆盖音频缓冲区上限、语音分段、字幕修订、翻译队列、翻译质量防护、
悬浮窗尺寸和程序停止清理等行为。

## 已知限制

- 嘈杂音频、多人重叠讲话、口音和背景音乐可能导致识别错误。
- 小型本地翻译模型的表达不一定像大型云端模型一样自然。
- 临时字幕会随句子继续变化，这是实时修订机制的正常行为。
- CPU 模式可能跟不上快速连续讲话。
- 当前没有商业代码签名；自行构建的 Windows 程序可能触发未知发布者提示。

## 仓库与模型许可

本仓库自己的代码采用 [MIT License](LICENSE)。模型权重和第三方运行库不属于本项目，
也不会提交到本仓库；详细来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 贡献与问题反馈

提交问题时请说明 Windows 版本、CPU、显卡、显存、视频语言、音频输出设备以及大致延迟。
请不要上传含有私人对话的原始录音。
