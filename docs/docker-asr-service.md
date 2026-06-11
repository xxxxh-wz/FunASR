# Docker ASR 服务部署与字幕 API

本文档说明如何把当前本地批处理听译 pipeline 部署为 Docker GPU 服务，并通过 API 生成字幕文件。

## 构建与启动

```bash
docker compose -f docker-compose.asr.yml build
docker compose -f docker-compose.asr.yml up asr-server
```

默认挂载：

- `/home/dell/models:/models`
- `./音频文件:/data/input`
- `./听译结果:/data/output`

默认服务端 route 覆盖 `fun-asr-nano-vllm`、`qwen3-asr`、`qwen3-asr-vllm`。生产优先使用 `qwen3-asr-vllm`。

ModelScope 本地路径会把模型名中的小数点转义，例如 Qwen3-ASR 1.7B 的默认挂载路径是 `/models/Qwen/Qwen3-ASR-1___7B`。

## 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

返回内容包含支持的 routes、模型路径和当前模型加载状态。

## 字幕文件 API

`POST /asr/subtitles` 接收一个或多个音频文件，返回 zip 包。zip 内包含与上传文件同名的 `.md`、`.srt`、`.json` 文件；单个文件失败时，会生成对应 `.error.json`。

```bash
curl -X POST http://127.0.0.1:8000/asr/subtitles \
  -F "model=qwen3-asr-vllm" \
  -F "language=zh" \
  -F "timestamps=true" \
  -F "speaker_diarization=true" \
  -F "output_formats=md,srt,json" \
  -F "hotwords=约当块,拉姆达,最小多项式" \
  -F "files=@./音频文件/test.m4a" \
  -o ./听译结果/subtitles.zip
```

保留的 JSON 批处理接口：

```bash
curl -X POST http://127.0.0.1:8000/asr/batch \
  -F "model=qwen3-asr-vllm" \
  -F "language=zh" \
  -F "timestamps=true" \
  -F "speaker_diarization=true" \
  -F "files=@./音频文件/test.m4a"
```

## 容器内 CLI 批处理

服务启动后，也可以使用同一镜像运行目录批处理客户端：

```bash
docker compose -f docker-compose.asr.yml run --rm asr-server \
  uv run batch-transcribe \
  --input-dir /data/input \
  --output-dir /data/output \
  --server-url http://asr-server:8000 \
  --route qwen3-asr-vllm \
  --batch-size 2 \
  --language zh \
  --hotwords "约当块,拉姆达,最小多项式"
```

如果在宿主机运行 CLI，则把 `--server-url` 改为 `http://127.0.0.1:8000`。
