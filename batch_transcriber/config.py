from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


DEFAULT_AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
)


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path = Path("./音频文件")
    output_dir: Path = Path("./听译结果")
    recursive: bool = True
    overwrite: bool = False
    server_url: str = "http://127.0.0.1:8000"
    route: str = "fun-asr-nano-vllm"
    batch_size: int = 2
    concurrency: int = 1
    language: str = "auto"
    timestamps: bool = True
    speaker_diarization: bool = True
    timestamp_granularity: str = "sentence"
    translate: bool = False
    timeout_seconds: int = 1800
    retries: int = 2
    fail_fast: bool = False
    model_hub: str = "ms"
    model_cache_dir: Path = Path("/home/dell/models")
    audio_extensions: tuple[str, ...] = field(default_factory=lambda: DEFAULT_AUDIO_EXTENSIONS)
    retry_failed: bool = False
    merge_short_segments: bool = True
    min_merged_segment_seconds: float = 15.0
    target_merged_segment_seconds: float = 30.0
    max_merged_segment_seconds: float = 60.0
    max_merge_gap_seconds: float = 8.0
    min_speaker_turn_seconds: float = 8.0
    drop_filler_only_segments: bool = False
    hotwords: tuple[str, ...] = ()
    hotword_file: Path | None = None
    max_hotwords: int = 200
    max_hotword_chars: int = 64
    hotword_prompt_template: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BatchConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "BatchConfig":
        normalized = dict(data)
        for key in ("input_dir", "output_dir", "model_cache_dir", "hotword_file"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = Path(normalized[key])
        if "hotwords" in normalized and normalized["hotwords"] is not None:
            value = normalized["hotwords"]
            normalized["hotwords"] = (value,) if isinstance(value, str) else tuple(value)
        if "audio_extensions" in normalized and normalized["audio_extensions"] is not None:
            normalized["audio_extensions"] = tuple(
                ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
                for ext in normalized["audio_extensions"]
            )
        return cls(**normalized)

    def with_overrides(self, **kwargs: Any) -> "BatchConfig":
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        for key in ("input_dir", "output_dir", "model_cache_dir", "hotword_file"):
            if key in cleaned:
                cleaned[key] = Path(cleaned[key])
        if "hotwords" in cleaned and cleaned["hotwords"] is not None:
            value = cleaned["hotwords"]
            cleaned["hotwords"] = (value,) if isinstance(value, str) else tuple(value)
        return replace(self, **cleaned)

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.timestamp_granularity != "sentence":
            raise ValueError("timestamp_granularity must be sentence")
        if self.translate:
            raise ValueError("translation is out of scope for the current stage")
        if self.min_merged_segment_seconds < 0:
            raise ValueError("min_merged_segment_seconds must be >= 0")
        if self.target_merged_segment_seconds <= 0:
            raise ValueError("target_merged_segment_seconds must be > 0")
        if self.max_merged_segment_seconds <= 0:
            raise ValueError("max_merged_segment_seconds must be > 0")
        if self.max_merged_segment_seconds < self.min_merged_segment_seconds:
            raise ValueError("max_merged_segment_seconds must be >= min_merged_segment_seconds")
        if self.max_merge_gap_seconds < 0:
            raise ValueError("max_merge_gap_seconds must be >= 0")
        if self.min_speaker_turn_seconds < 0:
            raise ValueError("min_speaker_turn_seconds must be >= 0")
        if self.max_hotwords < 0:
            raise ValueError("max_hotwords must be >= 0")
        if self.max_hotword_chars <= 0:
            raise ValueError("max_hotword_chars must be > 0")
