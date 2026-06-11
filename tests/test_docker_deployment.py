from pathlib import Path

import yaml


def test_asr_dockerfile_uses_uv_and_server_entrypoint():
    dockerfile = Path("Dockerfile.asr").read_text(encoding="utf-8")

    assert "nvidia/cuda" in dockerfile
    assert "UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/" in dockerfile
    assert "README.md" in dockerfile
    assert "python3.10-dev" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/healthz" in dockerfile


def test_asr_compose_mounts_models_audio_outputs_and_gpu():
    compose = yaml.safe_load(Path("docker-compose.asr.yml").read_text(encoding="utf-8"))
    service = compose["services"]["asr-server"]
    command = [str(item) for item in service["command"]]

    assert "/home/dell/models:/models" in service["volumes"]
    assert "./音频文件:/data/input" in service["volumes"]
    assert "./听译结果:/data/output" in service["volumes"]
    assert service["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "qwen3-asr-vllm" not in command
    assert "--qwen-model" in command
    assert "/models/Qwen/Qwen3-ASR-1___7B" in command
    assert "--qwen-gpu-memory-utilization" in command
    gpu_device = service["deploy"]["resources"]["reservations"]["devices"][0]
    assert gpu_device["device_ids"] == ["${FUNASR_GPU_DEVICE:-0}"]
    assert gpu_device["capabilities"] == ["gpu"]
