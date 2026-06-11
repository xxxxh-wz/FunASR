# 本地化批处理听译命令行工具需求文档

## 1. 背景与目标

基于 FunASR 项目提供的模型、VAD、vLLM 与本地服务部署能力，实现一个面向本地化推理的命令行批处理听译工具。

工具从 `./音频文件` 递归扫描音频文件，将音频按服务端 batch 方式提交到部署在 A100 服务器上的本地推理服务，由服务端负责 VAD 切分、无效音频过滤和模型推理。工具最终在 `./听译结果` 下生成与输入目录层级对齐的 `.md` 听译文档和 `.srt` 字幕文件。

当前阶段只做语音转文字，不做翻译。

输出后处理应按课堂转写稿粒度组织内容，而不是逐字逐句碎片化字幕。默认对服务端返回的句级 segment 做短段合并和说话人稳定化：最小段落 15 秒，目标段落约 30 秒，最大段落 60 秒；小于 8 秒的短暂说话人切换视为误分并入主讲；超过 8 秒的静音间隔不跨段合并。

## 2. 核心结论

本阶段不预设最终生产模型，先按四条路线做同一批音频的可复现实测，再根据准确率、速度、显存、稳定性、时间戳和批处理能力定型。

测试路线：

| 路线 | 模型/引擎 | 目标 | 当前判断 |
|---|---|---|---|
| A | `fun-asr-nano` | 基线准确率、句级时间戳、服务端 VAD | 必测 |
| B | `fun-asr-nano + vLLM` | 验证服务端 batch 吞吐和 A100 利用率 | 必测，优先服务化 |
| C | `Qwen3-ASR` | 验证 1.7B 大模型准确率、多语言能力 | 必测 |
| D | `Qwen3-ASR + vLLM` | 验证更大模型在 vLLM 下的可行性和吞吐 | 探索项，需先确认 FunASR/Qwen3-ASR 当前 vLLM 接入方式 |

为什么不直接默认 `FunAudioLLM/Fun-ASR-Nano-2512`：

- `fun-asr-nano` 在 FunASR 内已有完整 vLLM、服务端 VAD、批量分段推理和服务示例，工程落地路径更短。
- `Qwen3-ASR-1.7B` 参数量更大，FunASR 已支持 AutoModel 方式调用，并支持 ModelScope 下载；理论上应纳入精度优先路线。
- 生产选型不能只看参数量，需要同时比较：目标音频上的字错率/人工抽检质量、句级时间戳质量、长音频稳定性、A100 显存、RTF、服务端 batch 能力、失败率和部署复杂度。
- 因此本需求将模型选型改为“测试路线”，最终默认模型由 benchmark 结果决定。

## 3. 环境与模型下载

### 3.1 环境管理

项目环境使用 `uv` 管理。

建议要求：

- 使用 `uv venv` 创建虚拟环境。
- 使用 `uv pip install` 安装依赖。
- 如后续新增独立工具包，应维护 `pyproject.toml` 和 `uv.lock`。
- 服务端和客户端环境可以分离，但依赖版本必须在文档中记录。

基础依赖建议：

```bash
uv venv
source .venv/bin/activate
uv pip install -e . --default-index https://mirrors.aliyun.com/pypi/simple/
uv pip install modelscope fastapi uvicorn python-multipart soundfile librosa rich pyyaml httpx
uv pip install qwen-asr
uv pip install "vllm==0.11.2"
```

本项目已在 uv 隔离环境内验证 `vllm==0.11.2`、`torch==2.9.0+cu128` 可在当前 A800/A100 类服务器上正常调用 CUDA。后续如升级 CUDA、PyTorch 或 vLLM，必须先通过同一套 smoke test 和整文件回归测试。

`pyproject.toml` 应配置 uv 默认使用阿里云镜像源：

```toml
[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
default = true
```

### 3.2 ModelScope 下载与本地模型目录

所有模型优先从 ModelScope 下载，调用时显式使用：

```python
hub="ms"
```

模型下载和缓存目录统一为：

```bash
/home/dell/models/
```

实现要求：

- 优先通过 ModelScope 下载。
- 模型文件必须落到 `/home/dell/models/`。
- 服务启动时优先使用本地模型路径，避免每次启动重复下载。
- benchmark 记录中区分“首次下载耗时”“模型加载耗时”“稳定推理耗时”。

建议实现方式：

```python
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-ASR-1.7B",
    cache_dir="/home/dell/models",
)
```

也可在服务启动脚本中设置：

```bash
export MODELSCOPE_CACHE=/home/dell/models
```

### 3.3 候选模型

候选模型：

| 别名 | ModelScope/HF ID | 说明 |
|---|---|---|
| `fun-asr-nano` | `FunAudioLLM/Fun-ASR-Nano-2512` | LLM-based ASR，31 语种，已有 FunASR vLLM 服务路线 |
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` | Qwen3-ASR 大模型，FunASR AutoModel 已支持 |
| `qwen3-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B` | Qwen3-ASR 轻量规格，可作为资源受限备选 |
| `fsmn-vad` | `fsmn-vad` | 服务端 VAD |
| `spk` | `iic/speech_eres2netv2_sv_zh-cn_16k-common` 或 `cam++` | 多说话人能力 |

Qwen3-ASR ModelScope 调用示例：

```python
from funasr import AutoModel

model = AutoModel(
    model="Qwen/Qwen3-ASR-1.7B",
    hub="ms",
    device="cuda:0",
    dtype="bf16",
)

res = model.generate(
    input="audio.wav",
    language="Chinese",
)
```

Fun-ASR-Nano vLLM 调用示例：

```python
from funasr.auto.auto_model_vllm import AutoModelVLLM

model = AutoModelVLLM(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
)

results = model.generate(["audio1.wav", "audio2.wav"], language="中文")
```

## 4. Benchmark 与模型选型要求

### 4.1 测试集

测试集来自：

```bash
./音频文件
```

要求：

- 递归扫描全部音频。
- 至少抽取一批短音频、中等音频、长音频分别测试。
- 如包含会议、访谈、多人对话、噪声、空白段，需要在测试报告中标注。
- 如无人工标注文本，则先用人工抽检评分；如有参考文本，则计算 CER/WER。

### 4.2 四条路线

路线 A：`fun-asr-nano`

- 使用 `FunAudioLLM/Fun-ASR-Nano-2512`。
- 使用 `hub="ms"`。
- 使用服务端 `fsmn-vad` 切分。
- 记录准确率、句级时间戳、处理耗时、显存和失败率。

路线 B：`fun-asr-nano + vLLM`

- 使用 FunASR vLLM 服务路线。
- 服务端支持 VAD 后对有效语音段做 batch 推理。
- 优先用于验证 A100 吞吐。
- 重点记录 batch size 对吞吐、显存和失败率的影响。

路线 C：`Qwen3-ASR`

- 使用 `Qwen/Qwen3-ASR-1.7B`。
- 使用 `hub="ms"`。
- 使用 `dtype="bf16"`。
- 需要安装 `qwen-asr`。
- VAD 仍由服务端统一处理：先切分有效语音段，再调用 Qwen3-ASR。
- 重点比较识别质量是否显著优于 `fun-asr-nano`。

路线 D：`Qwen3-ASR + vLLM`

- 作为探索路线。
- 先确认当前 FunASR/Qwen3-ASR 是否已有稳定 vLLM 接口。
- 如果没有现成接口，需要单独评估改造成本，不阻塞 A/B/C 三条路线。
- 不应在未验证前作为第一期生产依赖。

### 4.3 指标

每条路线必须记录：

- 模型名称、模型 revision、本地模型路径。
- FunASR、PyTorch、CUDA、vLLM、qwen-asr 版本。
- A100 型号、显存、GPU 数量、tensor parallel 配置。
- batch size。
- 音频总时长。
- 模型首次下载耗时。
- 模型加载耗时。
- 推理耗时。
- RTF 或 x realtime。
- 峰值显存。
- 成功数、失败数、失败原因。
- 句级时间戳是否稳定。
- 多说话人结果是否可用。
- 人工抽检结论或 CER/WER。

## 5. 推理服务需求

### 5.1 服务端职责

服务端部署在 A100 服务器上，负责：

- 加载模型。
- 从 ModelScope 下载或读取 `/home/dell/models/` 下的本地模型。
- 接收客户端一次请求内提交的多个音频文件。
- 对每个音频执行 VAD，截断/过滤无效音频段。
- 对 VAD 后的有效语音段进行模型推理。
- 聚合每个原始音频的句级结果。
- 可选输出多说话人标签。
- 返回结构化 JSON。

VAD 必须在服务端处理，客户端不做音频切分。

### 5.2 服务端 batch size 定义

本需求中的 `batch size` 指服务端一次请求内处理的音频文件数量，不是客户端并发请求数。

示例：

```bash
--batch-size 4
```

含义：

- 客户端每次从扫描结果中取 4 个音频文件。
- 通过一个 batch 请求提交给服务端。
- 服务端在同一次请求中完成这 4 个文件的 VAD、推理和结果聚合。

要求：

- 第一版可以先实现单 batch 串行请求，即客户端一次只发一个 batch 请求。
- 后续如需要进一步压榨吞吐，可增加 `--concurrency` 控制多个 batch 请求并发，但该参数不等同于 `batch size`。
- 服务端必须限制最大 batch size，并在超限时返回明确错误。

### 5.3 API 设计

OpenAI 兼容 `/v1/audio/transcriptions` 适合单文件转写，但无法自然表达“一次请求内多个音频文件”的 batch size。为满足本需求，建议新增或封装服务端批处理接口。

建议接口：

```http
POST /asr/batch
Content-Type: multipart/form-data
```

请求字段：

| 字段 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `files` | 是 | 无 | 多个音频文件 |
| `model` | 否 | benchmark 当前路线 | 模型别名 |
| `language` | 否 | `auto` | 语言提示 |
| `timestamps` | 否 | `true` | 输出句级时间戳 |
| `speaker_diarization` | 否 | `true` | 输出说话人 |
| `output_granularity` | 否 | `sentence` | 时间戳粒度，本阶段固定句级 |

响应示例：

```json
{
  "model": "fun-asr-nano-vllm",
  "batch_size": 2,
  "results": [
    {
      "file_name": "会议/会议录音01.wav",
      "status": "success",
      "duration": 1938.2,
      "processing_time": 222.1,
      "segments": [
        {
          "id": 1,
          "start": 0.42,
          "end": 3.8,
          "speaker": "SPK0",
          "text": "我们先讨论今天的议程。"
        }
      ],
      "text": "我们先讨论今天的议程。"
    },
    {
      "file_name": "访谈/访谈03.mp3",
      "status": "failed",
      "error": "decode failed"
    }
  ]
}
```

兼容要求：

- 如第一阶段暂未实现 `/asr/batch`，可临时由客户端循环调用单文件 `/asr` 或 `/v1/audio/transcriptions`，但文档和进度展示必须明确该模式不满足本需求中 batch size 的最终语义。
- `fun-asr-nano + vLLM` 可优先复用现有 `serve_vllm.py` 中的 `/asr` 逻辑，再扩展多文件 batch。
- `Qwen3-ASR` 路线需要实现同样的服务端 VAD + batch 聚合接口。

## 6. 命令行工具需求

### 6.1 输入扫描

默认输入目录：

```bash
./音频文件
```

扫描规则：

- 默认递归扫描子目录。
- 支持关闭递归扫描，但本阶段默认必须递归。
- 忽略隐藏文件、临时文件和空文件。
- 默认支持扩展名：

```text
.wav, .mp3, .flac, .m4a, .aac, .ogg, .opus, .webm, .mp4, .mov, .mkv
```

### 6.2 输出目录与目录层级

默认输出目录：

```bash
./听译结果
```

输出必须对齐 `./音频文件` 的相对目录层级。

示例：

```text
./音频文件/会议/2026/会议录音01.wav
./听译结果/会议/2026/会议录音01.md
./听译结果/会议/2026/会议录音01.srt
```

已存在输出文件时：

- 默认跳过 `.md` 和 `.srt` 都存在且非空的音频。
- 如果只存在其中一个文件，默认重新生成缺失文件。
- `--overwrite` 开启后覆盖已有输出。

### 6.3 输出格式

每个音频必须同时输出：

- `.md`：面向阅读、审阅和归档。
- `.srt`：面向视频字幕和播放器导入。

Markdown 示例：

```markdown
# 会议录音01

- 源文件：./音频文件/会议/2026/会议录音01.wav
- 输出文件：./听译结果/会议/2026/会议录音01.md
- 模型路线：fun-asr-nano-vllm
- 模型路径：/home/dell/models/FunAudioLLM/Fun-ASR-Nano-2512
- 语言：auto
- 是否翻译：否
- VAD：服务端开启
- 时间戳粒度：句级
- 多说话人：开启
- 音频时长：00:32:18
- 处理耗时：00:03:42
- 生成时间：2026-06-10 14:30:00

## 字幕

| 序号 | 时间 | 说话人 | 内容 |
|---:|---|---|---|
| 1 | 00:00:00.420 --> 00:00:03.800 | SPK0 | 我们先讨论今天的议程。 |
| 2 | 00:00:04.200 --> 00:00:07.100 | SPK1 | 好的，我这里有三个问题。 |
```

SRT 示例：

```srt
1
00:00:00,420 --> 00:00:03,800
[SPK0] 我们先讨论今天的议程。

2
00:00:04,200 --> 00:00:07,100
[SPK1] 好的，我这里有三个问题。
```

格式要求：

- 时间戳粒度为句级。
- `.md` 表格内容中的 `|` 必须转义为 `\|`。
- `.srt` 使用 `HH:MM:SS,mmm`。
- 若某段缺少说话人，则说话人列为空，SRT 不添加 `[SPKx]` 前缀。
- 若某文件无有效语音，仍生成 `.md`，说明“服务端 VAD 未检测到有效语音”；`.srt` 可为空或只写注释，具体实现需固定。

### 6.4 时间戳与多说话人

本阶段要求：

- 时间戳粒度：句级。
- 默认开启时间戳。
- 默认开启多说话人。
- 服务端返回的字级/词级时间戳不直接输出为主格式，可作为调试字段保留在可选 JSON 中。
- 如模型路线不支持时间戳或说话人，必须在 benchmark 和输出元信息中标记。

### 6.5 命令行参数

基础用法：

```bash
uv run python batch_transcribe.py
```

指定服务与 batch：

```bash
uv run python batch_transcribe.py \
  --input-dir ./音频文件 \
  --output-dir ./听译结果 \
  --server-url http://a100-server:8000 \
  --route fun-asr-nano-vllm \
  --batch-size 4
```

建议参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-dir` | `./音频文件` | 输入音频目录 |
| `--output-dir` | `./听译结果` | 输出目录 |
| `--recursive` | 开启 | 递归扫描 |
| `--batch-size` | `2` | 服务端一次请求内音频文件数量 |
| `--concurrency` | `1` | 客户端同时提交的 batch 请求数，默认不并发 |
| `--server-url` | `http://127.0.0.1:8000` | A100 推理服务地址 |
| `--route` | `fun-asr-nano-vllm` | 测试路线或生产路线 |
| `--language` | `auto` | 语言提示 |
| `--timestamps` | 开启 | 输出句级时间戳 |
| `--no-timestamps` | 关闭 | 不输出时间戳 |
| `--speaker-diarization` | 开启 | 输出说话人标签 |
| `--no-speaker-diarization` | 关闭 | 不输出说话人标签 |
| `--timeout` | `1800` | 单 batch 请求超时秒数 |
| `--retries` | `2` | 单 batch 失败重试次数 |
| `--overwrite` | 关闭 | 覆盖已有输出 |
| `--fail-fast` | 关闭 | 遇到失败立即停止 |
| `--config` | `batch_transcribe.yaml` | 配置文件路径 |
| `--retry-failed` | 关闭 | 只处理失败清单中的文件 |

## 7. 配置文件

默认配置文件：

```text
batch_transcribe.yaml
```

示例：

```yaml
input_dir: "./音频文件"
output_dir: "./听译结果"
recursive: true
overwrite: false

server_url: "http://127.0.0.1:8000"
route: "fun-asr-nano-vllm"
batch_size: 2
concurrency: 1
language: "auto"
timestamps: true
speaker_diarization: true
timestamp_granularity: "sentence"
translate: false

timeout_seconds: 1800
retries: 2
fail_fast: false

model_hub: "ms"
model_cache_dir: "/home/dell/models"

audio_extensions:
  - ".wav"
  - ".mp3"
  - ".flac"
  - ".m4a"
  - ".aac"
  - ".ogg"
  - ".opus"
  - ".webm"
  - ".mp4"
  - ".mov"
  - ".mkv"
```

命令行参数优先级高于配置文件。

## 8. 进度展示

终端必须展示整体进度、batch 进度、单文件状态、耗时和失败原因。

整体进度：

- 总文件数。
- 已完成文件数。
- 成功数。
- 跳过数。
- 失败数。
- 当前 batch 序号。
- batch size。
- 总耗时。
- 预计剩余时间，能估算则展示。

batch 进度：

- 当前 batch 包含的文件数。
- batch 上传中、服务端 VAD 中、服务端推理中、结果写入中。
- batch 耗时。
- batch 失败原因。

单文件状态：

- 等待。
- 已提交。
- 服务端处理中。
- 写入 `.md`。
- 写入 `.srt`。
- 成功。
- 失败。
- 跳过。

示例：

```text
[整体] 12/80 完成 | 成功 10 | 跳过 1 | 失败 1 | batch_size 4 | 用时 00:08:31 | ETA 00:42:10
[Batch 4] 4 files | 服务端推理中 | 已用时 00:01:42
[成功] 会议/2026/会议录音01.wav -> 听译结果/会议/2026/会议录音01.md, .srt | 00:03:42
[失败] 访谈/访谈03.mp3 | HTTP 500: CUDA out of memory | 已重试 2 次
```

建议使用 `rich`；非交互终端自动降级为普通文本日志。

## 9. 日志、失败清单与断点续跑

输出目录下生成：

```text
./听译结果/transcription_batch.log
./听译结果/failed_files.jsonl
./听译结果/benchmark_results.jsonl
```

失败记录示例：

```json
{
  "file": "./音频文件/访谈/访谈03.mp3",
  "output_md": "./听译结果/访谈/访谈03.md",
  "output_srt": "./听译结果/访谈/访谈03.srt",
  "route": "qwen3-asr-1.7b",
  "batch_size": 4,
  "server_url": "http://a100-server:8000",
  "attempts": 3,
  "error_type": "http_error",
  "error_message": "HTTP 500: CUDA out of memory",
  "elapsed_seconds": 94.2,
  "time": "2026-06-10T14:35:20+08:00"
}
```

断点续跑：

- `.md` 和 `.srt` 都存在且非空时默认跳过。
- `--overwrite` 强制覆盖。
- `--retry-failed` 根据 `failed_files.jsonl` 重跑失败文件。
- 写文件必须先写临时文件，再原子替换，避免半成品被误判为成功。

## 10. 非功能需求

可靠性：

- 单文件失败不影响同 batch 其他文件的结果落盘。
- 单 batch 失败后按配置重试。
- 服务端返回部分成功时，客户端必须保存成功文件并记录失败文件。
- 客户端中断后可续跑。

性能：

- batch size 可配置并受服务端上限保护。
- 服务端 VAD 过滤无效音频，避免无效段进入大模型推理。
- 大文件上传使用文件句柄，避免客户端一次性读入内存。
- benchmark 必须排除首次下载时间后再统计稳定吞吐。

兼容性：

- Linux + A100 优先。
- Python 环境由 `uv` 管理。
- 音频解码能力以服务端 `soundfile`/`ffmpeg`/`librosa` 支持为准。

安全：

- 默认只连接内网 A100 服务。
- 不上传公网服务。
- 如后续增加鉴权，客户端支持 `--api-key` 或环境变量读取。
- 日志不得输出密钥。

## 11. 验收标准

基础验收：

- `./音频文件` 下多级子目录中的音频可被递归扫描。
- 输出文件生成到 `./听译结果`，并保持相对目录层级。
- 每个成功音频同时生成 `.md` 和 `.srt`。
- 当前阶段不产生翻译文本。
- `--batch-size 4` 表示一次请求提交 4 个音频给服务端。
- 服务端 VAD 开启，客户端不做切分。

模型路线验收：

- 完成 `fun-asr-nano`、`fun-asr-nano + vLLM`、`Qwen3-ASR` 三条必测路线的 benchmark。
- `Qwen3-ASR + vLLM` 至少完成可行性结论：可用、需改造或暂不可用。
- 所有路线使用 `hub="ms"` 或本地 `/home/dell/models/` 路径。
- benchmark 记录模型路径、版本、显存、速度、失败率和质量结论。

输出验收：

- `.md` 可正常预览。
- `.srt` 可被常见播放器或字幕工具读取。
- 时间戳为句级。
- 多说话人开启时，输出说话人标签。
- 无有效语音文件不会导致任务崩溃。

进度验收：

- 终端可看到总进度、batch 进度、单文件状态、耗时和失败原因。
- 任务结束后输出成功、跳过、失败统计。

可靠性验收：

- 网络超时、HTTP 500、JSON 解析失败、磁盘写入失败均能记录。
- 重试次数符合配置。
- 中断后再次运行可跳过已完成文件并继续处理。

## 12. 实现建议

建议拆分为客户端 CLI 与服务端 ASR 两部分。

客户端模块：

```text
batch_transcriber/
  config.py
  scanner.py
  client.py
  formatter_md.py
  formatter_srt.py
  progress.py
  runner.py
```

服务端模块：

```text
asr_server/
  model_loader.py
  vad.py
  routes.py
  batch_api.py
  benchmark.py
```

第一期优先级：

1. 实现 `fun-asr-nano + vLLM` 服务端 batch 接口。
2. 实现客户端递归扫描、batch 提交、`.md`/`.srt` 输出。
3. 跑通 `./音频文件` 到 `./听译结果`。
4. 加入 `Qwen3-ASR` 普通推理路线。
5. 形成四路线 benchmark 报告。
6. 再决定最终生产默认模型。

## 13. 仍需确认

1. A100 服务具体地址、端口、GPU 数量和是否需要鉴权。——由于是在本机运行，地址为本机地址，且不需要鉴权。
2. `Qwen3-ASR + vLLM` 是否要求第一期必须生产可用，还是只需可行性评估。——只需要可行性评估，是否能用，以及是否需要改造。
3. 无有效语音文件的 `.srt` 输出格式：写一个 0 秒提示段。
4. 多说话人模型最终使用 `iic/speech_eres2netv2_sv_zh-cn_16k-common` 。
5. 是否需要额外输出 `.json` 作为调试和二次处理数据。——是
