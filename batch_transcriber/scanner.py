from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioTask:
    source_path: Path
    relative_path: Path
    output_md: Path
    output_srt: Path
    output_json: Path


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def is_complete(task: AudioTask) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in (task.output_md, task.output_srt, task.output_json))


def output_paths(input_dir: Path, output_dir: Path, audio_path: Path) -> AudioTask:
    rel = audio_path.relative_to(input_dir)
    stem_rel = rel.with_suffix("")
    return AudioTask(
        source_path=audio_path,
        relative_path=rel,
        output_md=output_dir / stem_rel.with_suffix(".md"),
        output_srt=output_dir / stem_rel.with_suffix(".srt"),
        output_json=output_dir / stem_rel.with_suffix(".json"),
    )


def discover_audio_files(
    input_dir: Path,
    output_dir: Path,
    extensions: tuple[str, ...],
    recursive: bool = True,
    overwrite: bool = False,
) -> list[AudioTask]:
    if not input_dir.exists():
        return []

    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    tasks: list[AudioTask] = []
    for path in sorted(iterator):
        if not path.is_file():
            continue
        rel = path.relative_to(input_dir)
        if _is_hidden(rel):
            continue
        if path.name.startswith("~") or path.stat().st_size == 0:
            continue
        if path.suffix.lower() not in normalized_exts:
            continue
        task = output_paths(input_dir, output_dir, path)
        if not overwrite and is_complete(task):
            continue
        tasks.append(task)
    return tasks


def chunk_tasks(tasks: list[AudioTask], batch_size: int) -> list[list[AudioTask]]:
    return [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]
