#!/usr/bin/env python3
"""Smoke test Qwen3-ASR through the qwen-asr vLLM backend."""

from __future__ import annotations

import argparse

from qwen_asr import Qwen3ASRModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timestamps", action="store_true")
    args = parser.parse_args()

    model = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    results = model.transcribe(
        audio=[args.audio],
        language=[args.language],
        return_time_stamps=args.timestamps,
    )
    print(type(results), len(results))
    for result in results:
        print(type(result))
        print("language", getattr(result, "language", None))
        print("text", getattr(result, "text", None))
        print("time_stamps", getattr(result, "time_stamps", None))


if __name__ == "__main__":
    main()
