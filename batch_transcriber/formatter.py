from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from batch_transcriber.config import BatchConfig
from batch_transcriber.scanner import AudioTask

FILLER_TEXTS = {"啊", "嗯", "哦", "呃", "对", "好", "是吧", "对吧"}
SENTENCE_ENDINGS = "。！？!?；;"


def _seconds(value: Any) -> float:
    if value is None:
        return 0.0
    value = float(value)
    return value / 1000 if value > 10000 else value


def format_duration(seconds: float | int | None) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_md_time(seconds: float | int | None) -> str:
    total_ms = max(0, int(round(float(seconds or 0) * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_srt_time(seconds: float | int | None) -> str:
    return format_md_time(seconds).replace(".", ",")


def escape_md_cell(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", "<br>")


def _segment_duration(segment: dict[str, Any]) -> float:
    return max(0.0, float(segment.get("end") or 0) - float(segment.get("start") or 0))


def _clean_text(text: Any) -> str:
    return str(text or "").strip()


def _is_filler_text(text: str) -> bool:
    cleaned = text.strip().strip(" 。？！!?，,.、")
    return cleaned in FILLER_TEXTS


def _join_text(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def normalize_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for index, segment in enumerate(result.get("segments") or [], 1):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "id": segment.get("id", index),
                "start": _seconds(segment.get("start")),
                "end": _seconds(segment.get("end")),
                "speaker": segment.get("speaker") or segment.get("spk") or "",
                "text": text,
            }
        )
    return normalized


def _stabilize_speakers(segments: list[dict[str, Any]], min_speaker_turn_seconds: float) -> list[dict[str, Any]]:
    if not segments or min_speaker_turn_seconds <= 0:
        return [dict(segment) for segment in segments]

    turns: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        speaker = segment.get("speaker") or ""
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end_index"] = index
            turns[-1]["duration"] += _segment_duration(segment)
        else:
            turns.append({"speaker": speaker, "start_index": index, "end_index": index, "duration": _segment_duration(segment)})

    stabilized = [dict(segment) for segment in segments]
    for turn_index, turn in enumerate(turns):
        speaker = turn["speaker"]
        if not speaker or turn["duration"] >= min_speaker_turn_seconds:
            continue

        previous_turn = turns[turn_index - 1] if turn_index > 0 else None
        next_turn = turns[turn_index + 1] if turn_index + 1 < len(turns) else None
        replacement = None
        if previous_turn and next_turn and previous_turn["speaker"] == next_turn["speaker"]:
            replacement = previous_turn["speaker"]
        elif previous_turn or next_turn:
            candidates = [candidate for candidate in (previous_turn, next_turn) if candidate and candidate["speaker"]]
            if candidates:
                replacement = max(candidates, key=lambda item: item["duration"])["speaker"]
        if not replacement:
            continue

        for segment_index in range(turn["start_index"], turn["end_index"] + 1):
            stabilized[segment_index]["speaker"] = replacement
    return stabilized


def _merge_segment_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    speaker_durations: dict[str, float] = {}
    for segment in group:
        speaker = segment.get("speaker") or ""
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + _segment_duration(segment)
    speaker = max(speaker_durations.items(), key=lambda item: item[1])[0] if speaker_durations else ""
    return {
        "id": group[0].get("id"),
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "speaker": speaker,
        "text": _join_text([segment["text"] for segment in group]),
    }


def _group_is_filler_only(group: list[dict[str, Any]]) -> bool:
    return bool(group) and all(_is_filler_text(segment["text"]) for segment in group)


def postprocess_segments(
    result: dict[str, Any],
    *,
    merge_short_segments: bool = True,
    min_segment_seconds: float = 15.0,
    target_segment_seconds: float = 30.0,
    max_segment_seconds: float = 60.0,
    max_merge_gap_seconds: float = 8.0,
    min_speaker_turn_seconds: float = 8.0,
    drop_filler_only_segments: bool = False,
) -> list[dict[str, Any]]:
    segments = normalize_segments(result)
    if not segments:
        return []

    segments = _stabilize_speakers(segments, min_speaker_turn_seconds)
    if drop_filler_only_segments:
        segments = [segment for segment in segments if not _is_filler_text(segment["text"])]
    if not merge_short_segments:
        return segments

    merged: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            if not _group_is_filler_only(current):
                merged.append(_merge_segment_group(current))
            current = []

    for segment in segments:
        if not current:
            current.append(segment)
            continue

        next_duration = float(segment["end"]) - float(current[0]["start"])
        current_duration = float(current[-1]["end"]) - float(current[0]["start"])
        current_text = _clean_text(current[-1]["text"])
        segment_text = _clean_text(segment["text"])
        same_speaker = (current[-1].get("speaker") or "") == (segment.get("speaker") or "")
        filler = _is_filler_text(segment_text) or _is_filler_text(current_text)
        gap = max(0.0, float(segment["start"]) - float(current[-1]["end"]))
        may_merge = same_speaker or current_duration < min_segment_seconds or filler

        if gap > max_merge_gap_seconds:
            flush()
            current.append(segment)
            continue
        if next_duration > max_segment_seconds:
            flush()
            current.append(segment)
            continue
        if current_duration >= target_segment_seconds and current_text.endswith(tuple(SENTENCE_ENDINGS)) and not filler:
            flush()
            current.append(segment)
            continue
        if may_merge:
            current.append(segment)
            continue

        flush()
        current.append(segment)

    flush()
    return merged


def _postprocess_config(config: BatchConfig | None) -> dict[str, Any]:
    if config is None:
        return {
            "merge_short_segments": True,
            "min_merged_segment_seconds": 15.0,
            "target_merged_segment_seconds": 30.0,
            "max_merged_segment_seconds": 60.0,
            "max_merge_gap_seconds": 8.0,
            "min_speaker_turn_seconds": 8.0,
            "drop_filler_only_segments": False,
        }
    return {
        "merge_short_segments": config.merge_short_segments,
        "min_merged_segment_seconds": config.min_merged_segment_seconds,
        "target_merged_segment_seconds": config.target_merged_segment_seconds,
        "max_merged_segment_seconds": config.max_merged_segment_seconds,
        "max_merge_gap_seconds": config.max_merge_gap_seconds,
        "min_speaker_turn_seconds": config.min_speaker_turn_seconds,
        "drop_filler_only_segments": config.drop_filler_only_segments,
    }


def processed_segments(result: dict[str, Any], config: BatchConfig | None = None) -> list[dict[str, Any]]:
    options = _postprocess_config(config)
    return postprocess_segments(
        result,
        merge_short_segments=bool(options["merge_short_segments"]),
        min_segment_seconds=float(options["min_merged_segment_seconds"]),
        target_segment_seconds=float(options["target_merged_segment_seconds"]),
        max_segment_seconds=float(options["max_merged_segment_seconds"]),
        max_merge_gap_seconds=float(options["max_merge_gap_seconds"]),
        min_speaker_turn_seconds=float(options["min_speaker_turn_seconds"]),
        drop_filler_only_segments=bool(options["drop_filler_only_segments"]),
    )


def render_markdown(
    task: AudioTask,
    result: dict[str, Any],
    route: str,
    language: str,
    started_at: datetime | None = None,
    config: BatchConfig | None = None,
) -> str:
    title = task.source_path.stem
    segments = processed_segments(result, config)
    generated = (started_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        "",
        f"- 源文件：{task.source_path}",
        f"- 输出文件：{task.output_md}",
        f"- 模型路线：{route}",
        f"- 模型路径：{result.get('model_path', '')}",
        f"- 语言：{language or 'auto'}",
        "- 是否翻译：否",
        "- VAD：服务端开启",
        "- 时间戳粒度：句级",
        f"- 多说话人：{'开启' if any(s['speaker'] for s in segments) else '未返回'}",
        f"- 音频时长：{format_duration(result.get('duration'))}",
        f"- 处理耗时：{format_duration(result.get('processing_time'))}",
        f"- 生成时间：{generated}",
        "",
        "## 字幕",
        "",
    ]
    if not segments:
        lines.append("服务端 VAD 未检测到有效语音。")
        lines.append("")
        return "\n".join(lines)

    lines.extend(["| 序号 | 时间 | 说话人 | 内容 |", "|---:|---|---|---|"])
    for index, segment in enumerate(segments, 1):
        time_range = f"{format_md_time(segment['start'])} --> {format_md_time(segment['end'])}"
        lines.append(
            f"| {index} | {time_range} | {escape_md_cell(segment['speaker'])} | {escape_md_cell(segment['text'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_srt(result: dict[str, Any], config: BatchConfig | None = None) -> str:
    segments = processed_segments(result, config)
    if not segments:
        return "1\n00:00:00,000 --> 00:00:00,000\n服务端 VAD 未检测到有效语音。\n"

    lines = []
    for index, segment in enumerate(segments, 1):
        text = segment["text"]
        if segment["speaker"]:
            text = f"[{segment['speaker']}] {text}"
        lines.extend(
            [
                str(index),
                f"{format_srt_time(segment['start'])} --> {format_srt_time(segment['end'])}",
                text,
                "",
            ]
        )
    return "\n".join(lines)


def render_json(task: AudioTask, result: dict[str, Any], route: str, language: str, config: BatchConfig | None = None) -> str:
    payload = {
        "source_file": str(task.source_path),
        "output_md": str(task.output_md),
        "output_srt": str(task.output_srt),
        "route": route,
        "language": language or "auto",
        "translate": False,
        "postprocess_config": _postprocess_config(config),
        "postprocessed_segments": processed_segments(result, config),
        "result": result,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
