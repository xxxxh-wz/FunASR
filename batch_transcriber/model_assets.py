from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_ALIASES = {
    "fun-asr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "fun-asr-nano-vllm": "FunAudioLLM/Fun-ASR-Nano-2512",
    "qwen3-asr": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-vllm": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-1.7b-vllm": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "spk-eres2netv2": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
}


DEFAULT_MODEL_ALIASES = ("fun-asr-nano", "fsmn-vad", "spk-eres2netv2")


@dataclass(frozen=True)
class ModelAsset:
    alias: str
    model_id: str
    local_path: Path


def resolve_model_id(alias_or_id: str) -> str:
    return MODEL_ALIASES.get(alias_or_id, alias_or_id)


def snapshot_download_model(model_id: str, cache_dir: str | Path) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    return snapshot_download(model_id, cache_dir=str(cache_dir))


def download_model(alias_or_id: str, cache_dir: str | Path = "/home/dell/models") -> ModelAsset:
    model_id = resolve_model_id(alias_or_id)
    local_path = snapshot_download_model(model_id, cache_dir=str(cache_dir))
    return ModelAsset(alias=alias_or_id, model_id=model_id, local_path=Path(local_path))


def download_models(aliases: list[str] | tuple[str, ...], cache_dir: str | Path = "/home/dell/models") -> list[ModelAsset]:
    return [download_model(alias, cache_dir=cache_dir) for alias in aliases]
