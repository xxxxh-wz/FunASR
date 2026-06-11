from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


HOTWORD_SPLIT_RE = re.compile(r"[,，、\n\r]+")


def _split_hotword_text(text: str) -> list[str]:
    return [part.strip() for part in HOTWORD_SPLIT_RE.split(text)]


def normalize_hotwords(
    sources: Iterable[str] | str | None,
    *,
    max_hotwords: int = 200,
    max_hotword_chars: int = 64,
) -> tuple[str, ...]:
    if sources is None:
        return ()
    if max_hotwords <= 0:
        return ()
    if isinstance(sources, str):
        iterable: Iterable[str] = (sources,)
    else:
        iterable = sources

    seen: set[str] = set()
    normalized: list[str] = []
    for source in iterable:
        if source is None:
            continue
        for item in _split_hotword_text(str(source)):
            if not item or item.startswith("#"):
                continue
            if len(item) > max_hotword_chars:
                continue
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
            if len(normalized) >= max_hotwords:
                return tuple(normalized)
    return tuple(normalized)


def load_hotwords(
    hotwords: Iterable[str] | str | None = None,
    *,
    hotword_file: str | Path | None = None,
    max_hotwords: int = 200,
    max_hotword_chars: int = 64,
) -> tuple[str, ...]:
    sources: list[str] = []
    if hotwords is not None:
        if isinstance(hotwords, str):
            sources.append(hotwords)
        else:
            sources.extend(str(item) for item in hotwords)
    if hotword_file is not None:
        sources.append(Path(hotword_file).read_text(encoding="utf-8"))
    return normalize_hotwords(sources, max_hotwords=max_hotwords, max_hotword_chars=max_hotword_chars)
