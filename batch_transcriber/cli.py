from __future__ import annotations

import argparse
from pathlib import Path

from batch_transcriber.config import BatchConfig
from batch_transcriber.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local batch transcription client for FunASR services.")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--route", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--timeout", type=int, default=None, dest="timeout_seconds")
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--retry-failed", action="store_true", default=None)
    parser.add_argument("--no-recursive", action="store_false", default=None, dest="recursive")
    parser.add_argument("--timestamps", action="store_true", default=None, dest="timestamps")
    parser.add_argument("--no-timestamps", action="store_false", dest="timestamps")
    parser.add_argument("--speaker-diarization", action="store_true", default=None, dest="speaker_diarization")
    parser.add_argument("--no-speaker-diarization", action="store_false", dest="speaker_diarization")
    parser.add_argument("--merge-short-segments", action="store_true", default=None, dest="merge_short_segments")
    parser.add_argument("--no-merge-short-segments", action="store_false", dest="merge_short_segments")
    parser.add_argument("--min-merged-segment-seconds", type=float, default=None)
    parser.add_argument("--target-merged-segment-seconds", type=float, default=None)
    parser.add_argument("--max-merged-segment-seconds", type=float, default=None)
    parser.add_argument("--max-merge-gap-seconds", type=float, default=None)
    parser.add_argument("--min-speaker-turn-seconds", type=float, default=None)
    parser.add_argument("--drop-filler-only-segments", action="store_true", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BatchConfig.from_yaml(args.config) if args.config else BatchConfig()
    overrides = vars(args).copy()
    overrides.pop("config", None)
    for key in ("input_dir", "output_dir"):
        if overrides.get(key) is not None:
            overrides[key] = Path(overrides[key])
    config = config.with_overrides(**overrides)
    summary = run(config)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
