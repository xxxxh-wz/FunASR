from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_transcriber.model_assets import DEFAULT_MODEL_ALIASES, MODEL_ALIASES, download_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download FunASR batch transcription models from ModelScope.")
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODEL_ALIASES), help="Model aliases or ModelScope IDs")
    parser.add_argument("--cache-dir", default="/home/dell/models", help="ModelScope cache directory")
    parser.add_argument("--list-aliases", action="store_true", help="List known aliases and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_aliases:
        print(json.dumps(MODEL_ALIASES, ensure_ascii=False, indent=2))
        return 0

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets = download_models(tuple(args.models), cache_dir=cache_dir)
    for asset in assets:
        print(f"{asset.alias}\t{asset.model_id}\t{asset.local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
