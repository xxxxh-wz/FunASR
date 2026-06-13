# Docker 部署踩坑以及使用指南

更新时间：2026-06-11

本文面向系统集成和接口调用同学，说明如何把本项目的本地听译能力部署为 Docker GPU 服务，并通过 HTTP API 生成 `.md`、`.srt`、`.json` 字幕/听译文件。

## 1. 推荐集成方式

推荐使用通用字幕生成接口：

```text
POST /asr/subtitles
```

原因：

- 一次请求可上传多个音频文件。
- 服务端完成 VAD、ASR、可选说话人识别、热词增强和字幕格式化。
- 返回 `application/zip`，ZIP 内直接包含可落盘的 `.md/.srt/.json` 文件。
- 单个文件失败时，不会导致整批结果完全不可用；失败文件会在 ZIP 内生成 `.error.json`。

保留调试接口：

```text
GET  /healthz
POST /asr/batch
POST /v1/audio/transcriptions
```

其中 `/asr/batch` 返回 JSON，适合联调和排查；`/v1/audio/transcriptions` 是 OpenAI Whisper 兼容入口，但当前字幕文件生成建议统一走 `/asr/subtitles`。

## 2. 机器与模型前置条件

已验证环境：

- GPU：A100
- CUDA 镜像：`nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- 模型缓存目录：`/home/dell/models`
- Python 依赖管理：`uv`
- PyPI 镜像：阿里云 `https://mirrors.aliyun.com/pypi/simple/`

模型目录要求：

| 模型 | 容器内路径 | 宿主机路径 |
|---|---|---|
| Fun-ASR-Nano | `/models/FunAudioLLM/Fun-ASR-Nano-2512` | `/home/dell/models/FunAudioLLM/Fun-ASR-Nano-2512` |
| Qwen3-ASR-1.7B | `/models/Qwen/Qwen3-ASR-1___7B` | `/home/dell/models/Qwen/Qwen3-ASR-1___7B` |
| VAD | `/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | `/home/dell/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| SPK | `/models/iic/speech_eres2netv2_sv_zh-cn_16k-common` | `/home/dell/models/iic/speech_eres2netv2_sv_zh-cn_16k-common` |

注意：ModelScope 会把模型名中的小数点转义为三个下划线。例如 `Qwen3-ASR-1.7B` 的本地目录是 `Qwen3-ASR-1___7B`。

## 3. 构建与启动

构建镜像：

```bash
docker compose -f docker-compose.asr.yml build
```

启动服务：

```bash
FUNASR_HOST_PORT=8903 FUNASR_GPU_DEVICE=0 \
docker compose -f docker-compose.asr.yml up -d --force-recreate --no-build asr-server
```

指定宿主机 GPU 3 时：

```bash
FUNASR_HOST_PORT=8903 FUNASR_GPU_DEVICE=3 \
docker compose -f docker-compose.asr.yml up -d --force-recreate --no-build asr-server
```

注意：`FUNASR_GPU_DEVICE` 选择宿主机 GPU；容器内固定使用 `CUDA_VISIBLE_DEVICES=0`。不要用 `CUDA_VISIBLE_DEVICES=3` 选择宿主机 GPU。

查看状态：

```bash
docker compose -f docker-compose.asr.yml ps
```

预期状态：

```text
STATUS: Up ... (healthy)
PORTS: 0.0.0.0:8903->8000/tcp
```

停止服务：

```bash
docker compose -f docker-compose.asr.yml down
```

默认挂载：

| 宿主机 | 容器内 | 用途 |
|---|---|---|
| `/home/dell/models` | `/models` | 模型缓存 |
| `./音频文件` | `/data/input` | 输入音频 |
| `./听译结果` | `/data/output` | 输出结果 |

## 4. 健康检查

```bash
curl http://127.0.0.1:8903/healthz
```

典型返回字段：

```json
{
  "status": "ok",
  "supported_routes": [
    "fun-asr-nano-vllm",
    "qwen3-asr",
    "qwen3-asr-vllm"
  ],
  "models_loaded": {
    "nano_vllm": true,
    "qwen3_asr": false,
    "qwen3_asr_vllm": false
  },
  "model_paths": {
    "nano": "/models/FunAudioLLM/Fun-ASR-Nano-2512",
    "qwen": "/models/Qwen/Qwen3-ASR-1___7B"
  }
}
```

说明：

- `qwen3-asr-vllm` 是懒加载。服务启动健康后，第一次请求该 route 时才加载 Qwen3-ASR vLLM engine。
- 第一次 Qwen 请求会比后续请求慢，接口调用方需要给首个请求更长超时时间。
- 新版服务会在 `/healthz` 的 `model_load_status` 中暴露模型加载状态。加载中时 `status` 会返回 `loading`，并给出 `loading_model`。
- 推荐在业务流量进入前显式预热 Qwen route，而不是让第一个字幕请求承担冷启动。

显式预热：

```bash
curl -X POST "http://127.0.0.1:8903/models/qwen3-asr-vllm/load?background=false"
curl http://127.0.0.1:8903/healthz
```

后台触发预热：

```bash
curl -X POST http://127.0.0.1:8903/models/qwen3-asr-vllm/load
```

Docker 启动时预加载：

```bash
FUNASR_HOST_PORT=8903 FUNASR_GPU_DEVICE=3 FUNASR_PRELOAD_MODELS=qwen3-asr-vllm \
docker compose -f docker-compose.asr.yml up -d --force-recreate --no-build asr-server
```

## 5. 字幕生成 API

### 5.1 请求

```text
POST /asr/subtitles
Content-Type: multipart/form-data
Accept: application/zip
```

表单字段：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `files` | file[] | 必填 | 一个或多个音频文件 |
| `model` | string | `qwen3-asr-vllm` | 推荐 `qwen3-asr-vllm`；可选 `fun-asr-nano-vllm`、`qwen3-asr` |
| `language` | string | 空 | 中文建议传 `zh` 或 `中文` |
| `hotwords` | string | 空 | 热词，逗号、空格或换行分隔 |
| `hotword_prompt_template` | string | 空 | Qwen route 的热词提示模板，通常不用传 |
| `speaker_diarization` | bool | `true` | 是否启用说话人识别 |
| `timestamps` | bool | `true` | 是否输出时间戳 |
| `output_granularity` | string | `sentence` | 当前字幕输出按句级段落 |
| `output_formats` | string | `md,srt,json` | 可选 `md`、`srt`、`json`，用逗号分隔 |

### 5.2 curl 示例

```bash
curl -X POST http://127.0.0.1:8903/asr/subtitles \
  -F "model=qwen3-asr-vllm" \
  -F "language=zh" \
  -F "timestamps=true" \
  -F "speaker_diarization=true" \
  -F "output_formats=md,srt,json" \
  -F "hotwords=约当块,拉姆达,最小多项式,矩阵多项式,导数" \
  -F "files=@./音频文件/test.m4a" \
  -o ./听译结果/subtitles.zip
```

多个文件：

```bash
curl -X POST http://127.0.0.1:8903/asr/subtitles \
  -F "model=qwen3-asr-vllm" \
  -F "language=zh" \
  -F "output_formats=md,srt,json" \
  -F "files=@./音频文件/a.m4a" \
  -F "files=@./音频文件/b.wav" \
  -o ./听译结果/subtitles.zip
```

### 5.3 Python requests 示例

```python
from pathlib import Path
import requests

url = "http://127.0.0.1:8903/asr/subtitles"
audio_paths = [Path("./音频文件/test.m4a")]

files = [("files", (p.name, p.open("rb"), "application/octet-stream")) for p in audio_paths]
data = {
    "model": "qwen3-asr-vllm",
    "language": "zh",
    "timestamps": "true",
    "speaker_diarization": "true",
    "output_formats": "md,srt,json",
    "hotwords": "约当块,拉姆达,最小多项式,矩阵多项式,导数",
}

try:
    resp = requests.post(url, data=data, files=files, timeout=1800)
    resp.raise_for_status()
    Path("./听译结果/subtitles.zip").write_bytes(resp.content)
finally:
    for _, file_tuple in files:
        file_tuple[1].close()
```

## 6. 返回 ZIP 结构

成功文件：

```text
subtitles.zip
├── test.md
├── test.srt
└── test.json
```

多文件会按上传文件名分别生成：

```text
subtitles.zip
├── a.md
├── a.srt
├── a.json
├── b.md
├── b.srt
└── b.json
```

单文件失败时：

```text
subtitles.zip
└── bad_file.error.json
```

`.error.json` 示例：

```json
{
  "file_name": "bad_file.m4a",
  "status": "failed",
  "error": "具体失败原因",
  "processing_time": 1.234,
  "hotwords_applied": true,
  "hotwords_count": 5
}
```

## 7. JSON 结果字段

`.json` 文件保留原始识别结构，核心字段包括：

| 字段 | 说明 |
|---|---|
| `text` | 全文 |
| `segments` | 句级片段数组 |
| `duration` | 音频时长，单位秒 |
| `processing_time` | 服务端处理耗时，单位秒 |
| `rtf` | `processing_time / duration` |
| `hotwords_applied` | 是否应用热词 |
| `hotwords_count` | 热词数量 |
| `model_path` | 实际模型路径 |
| `status` | `success` 或 `failed` |

`segments` 中常用字段：

```json
{
  "start": 0.98,
  "end": 60.99,
  "text": "识别文本",
  "speaker": "SPK0"
}
```

当前 `.md/.srt` 默认使用句级时间戳；如果启用 ForcedAligner，词级对齐会写入 JSON 的 `words` 字段，但不会覆盖默认句级字幕。

## 8. 已完成实际推理验证

Docker 服务实际测试配置：

- 服务地址：`http://127.0.0.1:8903`
- 接口：`POST /asr/subtitles`
- route：`qwen3-asr-vllm`
- 输入：课堂音频截取 120 秒 m4a
- 热词：`约当块, 拉姆达, 最小多项式, 矩阵多项式, 导数`
- 输出目录：`听译结果/docker_api_smoke/`

实际结果：

| 指标 | 结果 |
|---|---:|
| HTTP 状态 | 200 |
| 返回类型 | `application/zip` |
| 输出文件 | `.md/.srt/.json` |
| 音频时长 | 120.00s |
| 片段数 | 3 |
| 服务端处理耗时 | 57.862s |
| RTF | 0.4822 |
| 热词是否生效 | 是 |
| 热词数量 | 5 |

生成文件：

```text
听译结果/docker_api_smoke/subtitles.zip
听译结果/docker_api_smoke/docker_smoke.md
听译结果/docker_api_smoke/docker_smoke.srt
听译结果/docker_api_smoke/docker_smoke.json
```

`.srt` 已验证按句级时间戳输出，并带有说话人标签 `SPK0`。

## 9. 常见踩坑

### 9.1 Qwen 模型路径不是点号目录

错误路径：

```text
/models/Qwen/Qwen3-ASR-1.7B
```

正确路径：

```text
/models/Qwen/Qwen3-ASR-1___7B
```

如果路径错误，Qwen 首次请求会找不到 `config.json`。

### 9.2 Docker 镜像内需要复制项目元数据

`uv sync --frozen` 需要读取项目元数据。Dockerfile 里必须复制：

```text
README.md
LICENSE
MODEL_LICENSE
pyproject.toml
uv.lock
```

否则构建阶段会失败。

### 9.3 vLLM/Triton 运行期需要 Python 头文件

容器内需要安装：

```text
python3.10-dev
python3.11-dev
```

实际遇到过缺少 `/usr/include/python3.10/Python.h` 导致 Triton/vLLM 编译失败。

### 9.4 本地 SPK 模型路径不能直接当注册名

FunASR 的说话人模型加载需要注册模型名配合 `model_path`。当前服务端已做适配：

```text
model = iic/speech_eres2netv2_sv_zh-cn_16k-common
model_path = /models/iic/speech_eres2netv2_sv_zh-cn_16k-common
```

接口调用方无需处理，但部署时需要保证本地目录存在。

### 9.5 首次 Qwen 请求可能较慢

服务启动时先加载 Nano、VAD、SPK。`qwen3-asr-vllm` 在第一次请求时懒加载，因此：

- 健康检查通过不代表 Qwen engine 已经加载。
- 首次请求建议设置较长超时，例如 30 分钟。
- 后续 warm 请求耗时会明显下降。
- 如果业务侧直接用首个 `/asr/subtitles` 请求触发 Qwen 冷启动，可能出现长时间无输出、客户端超时后服务端才完成加载的体验。
- 推荐部署后调用 `POST /models/qwen3-asr-vllm/load?background=false` 完成预热，或设置 `FUNASR_PRELOAD_MODELS=qwen3-asr-vllm` 让容器启动阶段预加载。
- Docker 健康检查只判断 HTTP 服务是否可响应，不代表所有懒加载模型都 ready。以 `/healthz.model_load_status.qwen3_asr_vllm.status == "loaded"` 作为 Qwen ready 判据。

若观察到显存已占用、GPU 利用率为 0、CPU 单进程约 100%，通常表示服务卡在 CPU 侧的加载、调度、VAD/音频处理或 Python 后处理阶段，而不是 CUDA kernel 正在运行。此时优先查看：

```bash
docker logs --tail 500 funasr-asr-server-1
docker stats --no-stream funasr-asr-server-1
nvidia-smi
curl --max-time 10 http://127.0.0.1:8903/healthz
```

如果容器 `Up` 但 `/healthz` 连接拒绝或持续超时，应先重启服务，并用预热接口确认 Qwen 已加载后再放业务流量。

### 9.6 显存参数需要按服务形态调整

当前 Docker compose 使用偏保守配置：

```text
--gpu-memory-utilization 0.18
--qwen-gpu-memory-utilization 0.18
--qwen-max-model-len 8192
--qwen-max-inference-batch-size 16
--qwen-max-new-tokens 1024
--max-batch-size 8
```

这组参数适合稳定联调和短样本验证。生产压测时可逐步提高 batch 和 Qwen 显存利用率，但需要观察：

- GPU 剩余显存
- 首次加载是否失败
- 单请求耗时
- RTF
- 请求超时率

### 9.7 宿主机 GPU 选择不要用容器内 `CUDA_VISIBLE_DEVICES`

Docker compose 通过 `FUNASR_GPU_DEVICE` 选择宿主机 GPU：

```bash
FUNASR_HOST_PORT=8903 FUNASR_GPU_DEVICE=3 \
docker compose -f docker-compose.asr.yml up -d --force-recreate --no-build asr-server
```

容器内环境固定为：

```text
CUDA_VISIBLE_DEVICES=0
```

原因：当 Docker 只给容器分配 1 张 GPU 时，容器内的可见 GPU 会重新编号为 `cuda:0`。如果把 `CUDA_VISIBLE_DEVICES=3` 传进容器，PyTorch 会尝试寻找容器内第 4 张 GPU，最终报错：

```text
RuntimeError: No CUDA GPUs are available
```

### 9.8 不建议接口侧自行切 VAD

当前 pipeline 顺序是：

```text
原始音频 -> 服务端 VAD -> 有效语音段 -> ASR -> 可选 SPK/ForcedAligner -> 字幕文件
```

接口侧只需上传原始音频。自行切分可能破坏上下文，导致句级时间戳和说话人结果不一致。

### 9.9 热词要按课程或章节控制规模

`qwen3-asr-vllm` 使用模板化 context 承载热词，实际 A/B 中专业词改善明显。建议：

- 每次只传当前课程/章节相关热词。
- 优先传高频、易错、专业名词。
- 不要把完整教材术语库无差别塞进每个请求。

### 9.10 长音频 CPU 高、GPU 低不一定是异常

`/asr/subtitles` 是同步接口，返回 ZIP 前会依次经历：

```text
上传读取 -> 音频解码/重采样 -> VAD -> Qwen vLLM 推理 -> 可选 SPK -> JSON/MD/SRT 渲染 -> ZIP 写出
```

只有 `qwen_vllm_infer` 阶段会明显拉高 GPU SM 利用率；上传、解码、VAD、SPK 和 ZIP 阶段都可能表现为 CPU 高、GPU 低。长音频请求中，`curl` 显示 `100% upload` 只代表文件已传完，不代表服务端已完成转写。

服务端日志会输出阶段耗时，例如：

```text
ASR stage=audio_decode elapsed=0.079s file=xxx.m4a sr=48000 duration=60.011
ASR stage=qwen_vllm_vad elapsed=0.169s duration=60.011 segments=1
ASR stage=qwen_vllm_wait segments=1 timestamps=False
ASR stage=qwen_vllm_infer elapsed=4.080s segments=1
ASR stage=qwen_vllm_total elapsed=4.255s output_segments=1
ASR stage=zip_write elapsed=0.001s file=xxx.json bytes=4266
```

排查时优先看：

```bash
docker logs --tail 200 funasr-asr-server-1
nvidia-smi dmon -s pucm
docker stats --no-stream funasr-asr-server-1
```

如果业务侧会并发提交多个长音频请求，建议在接口前加队列。当前单 GPU 单 vLLM engine 的服务端会串行执行同一模型的推理调用，避免多个长请求同时进入同一个 engine 后互相抢 CPU 线程和 GPU 调度资源。

如果日志停在 `audio_decode`，没有继续出现 `qwen_vllm_vad` / `qwen_vllm_infer`，重点检查解码日志里的 `shape` 和 `channels`。从视频抽出的 m4a 常见为 `48000Hz + stereo`，服务端需要先降为单声道，再重采样到 16k；否则重采样库可能把声道维当作时间维处理，表现为 CPU 持续占用而 GPU 空闲。服务端已在预处理阶段按“单声道化 -> 16k 重采样 -> VAD”的顺序处理，并且 VAD / vLLM 共享模型调用会串行进入。

## 10. 接口侧建议

调用方建议实现：

- 上传前校验音频格式和大小。
- 请求超时区分 cold start 与 warm 请求。
- 将 ZIP 原样保存，再异步解压入业务目录。
- 检查 ZIP 中是否存在 `.error.json`。
- 记录 `processing_time`、`rtf`、`status`、`error`，便于后续统计。
- 对长音频或多文件批量请求设置任务队列，不建议同步 HTTP 请求无限等待。

推荐默认参数：

```text
model=qwen3-asr-vllm
language=zh
timestamps=true
speaker_diarization=true
output_granularity=sentence
output_formats=md,srt,json
```

如果只需要字幕文件，可把 `output_formats` 设为：

```text
md,srt
```

## 11. 联调验收清单

上线前至少验证：

- `GET /healthz` 返回 200。
- 单个 1 到 2 分钟音频可返回 ZIP。
- ZIP 中包含 `.md/.srt/.json`。
- `.srt` 时间戳可被播放器或字幕工具读取。
- 传入坏文件时 ZIP 中出现 `.error.json`。
- 热词传入后 `.json` 中 `hotwords_applied=true`。
- 首次 Qwen 请求和 warm 请求均不会超时。
- Docker 容器状态为 `healthy`。
