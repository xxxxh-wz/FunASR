from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from batch_transcriber.client import BatchTranscriptionClient, result_key
from batch_transcriber.config import BatchConfig
from batch_transcriber.formatter import atomic_write, render_json, render_markdown, render_srt
from batch_transcriber.hotwords import load_hotwords
from batch_transcriber.scanner import AudioTask, chunk_tasks, discover_audio_files


@dataclass
class RunSummary:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _failure_payload(task: AudioTask, config: BatchConfig, error: str, attempts: int, elapsed: float) -> dict[str, Any]:
    return {
        "file": str(task.source_path),
        "output_md": str(task.output_md),
        "output_srt": str(task.output_srt),
        "output_json": str(task.output_json),
        "route": config.route,
        "batch_size": config.batch_size,
        "server_url": config.server_url,
        "attempts": attempts,
        "error_type": "transcription_error",
        "error_message": error,
        "elapsed_seconds": round(elapsed, 3),
        "time": datetime.now(timezone.utc).isoformat(),
    }


def _write_success(task: AudioTask, result: dict[str, Any], config: BatchConfig) -> None:
    atomic_write(task.output_md, render_markdown(task, result, config.route, config.language, config=config))
    atomic_write(task.output_srt, render_srt(result, config=config))
    atomic_write(task.output_json, render_json(task, result, config.route, config.language, config=config))


def _load_retry_failed_tasks(config: BatchConfig) -> list[AudioTask]:
    failed_path = config.output_dir / "failed_files.jsonl"
    if not failed_path.exists():
        return []

    tasks = []
    seen = set()
    for line in failed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        file_path = Path(payload.get("file", ""))
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            relative = file_path.relative_to(config.input_dir)
        except ValueError:
            relative = Path(file_path.name)
        key = str(file_path)
        if key in seen:
            continue
        seen.add(key)
        stem_rel = relative.with_suffix("")
        tasks.append(
            AudioTask(
                source_path=file_path,
                relative_path=relative,
                output_md=Path(payload.get("output_md") or config.output_dir / stem_rel.with_suffix(".md")),
                output_srt=Path(payload.get("output_srt") or config.output_dir / stem_rel.with_suffix(".srt")),
                output_json=Path(payload.get("output_json") or config.output_dir / stem_rel.with_suffix(".json")),
            )
        )
    return tasks


def _match_results(tasks: list[AudioTask], payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_key = {result_key(item): item for item in payload["results"]}
    matched = {}
    for task in tasks:
        candidates = (task.relative_path.as_posix(), task.source_path.name, str(task.source_path))
        for candidate in candidates:
            if candidate in by_key:
                matched[task.relative_path.as_posix()] = by_key[candidate]
                break
    return matched


def run(config: BatchConfig, client: BatchTranscriptionClient | None = None) -> RunSummary:
    config.validate()
    hotwords = load_hotwords(
        config.hotwords,
        hotword_file=config.hotword_file,
        max_hotwords=config.max_hotwords,
        max_hotword_chars=config.max_hotword_chars,
    )
    if config.retry_failed:
        discovered = _load_retry_failed_tasks(config)
    else:
        discovered = discover_audio_files(
            config.input_dir,
            config.output_dir,
            config.audio_extensions,
            recursive=config.recursive,
            overwrite=config.overwrite,
        )
    summary = RunSummary(total=len(discovered))
    client = client or BatchTranscriptionClient(config.server_url, config.timeout_seconds)
    failed_path = config.output_dir / "failed_files.jsonl"
    benchmark_path = config.output_dir / "benchmark_results.jsonl"

    batches = chunk_tasks(discovered, config.batch_size)
    print(f"Route: {config.route}")
    print(f"Hotwords: {'enabled, ' + str(len(hotwords)) + ' terms' if hotwords else 'disabled'}")
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches, 1):
        batch_start = time.perf_counter()
        print(f"[Batch {batch_index}/{len(batches)}] {len(batch)} files | 上传中")
        last_error = None
        payload = None
        for attempt in range(1, config.retries + 2):
            try:
                payload = client.transcribe_batch(
                    batch,
                    route=config.route,
                    language=config.language,
                    timestamps=config.timestamps,
                    speaker_diarization=config.speaker_diarization,
                    hotwords=hotwords,
                    hotword_prompt_template=config.hotword_prompt_template,
                )
                break
            except Exception as exc:  # noqa: BLE001 - keep CLI robust and record exact failure
                last_error = str(exc)
                if attempt > config.retries:
                    break

        elapsed = time.perf_counter() - batch_start
        if payload is None:
            for task in batch:
                summary.failed += 1
                _append_jsonl(failed_path, _failure_payload(task, config, last_error or "unknown error", config.retries + 1, elapsed))
                print(f"[失败] {task.relative_path.as_posix()} | {last_error}")
            if config.fail_fast:
                break
            continue

        matched = _match_results(batch, payload)
        for task in batch:
            result = matched.get(task.relative_path.as_posix())
            if not result:
                summary.failed += 1
                error = "missing result in batch response"
                _append_jsonl(failed_path, _failure_payload(task, config, error, 1, elapsed))
                print(f"[失败] {task.relative_path.as_posix()} | {error}")
                if config.fail_fast:
                    return summary
                continue
            if result.get("status", "success") != "success":
                summary.failed += 1
                error = str(result.get("error") or "server returned failed status")
                _append_jsonl(failed_path, _failure_payload(task, config, error, 1, float(result.get("processing_time") or elapsed)))
                print(f"[失败] {task.relative_path.as_posix()} | {error}")
                if config.fail_fast:
                    return summary
                continue
            _write_success(task, result, config)
            summary.success += 1
            print(f"[成功] {task.relative_path.as_posix()} -> {task.output_md}, .srt, .json")

        _append_jsonl(
            benchmark_path,
            {
                "route": config.route,
                "batch_size": len(batch),
                "elapsed_seconds": round(elapsed, 3),
                "server_url": config.server_url,
                "hotwords_count": len(hotwords),
                "time": datetime.now(timezone.utc).isoformat(),
            },
        )
        total_elapsed = time.perf_counter() - started
        print(
            f"[整体] {summary.success + summary.failed}/{summary.total} 完成 | "
            f"成功 {summary.success} | 失败 {summary.failed} | 用时 {total_elapsed:.1f}s"
        )

    return summary
