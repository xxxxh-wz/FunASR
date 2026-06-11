from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from batch_transcriber.scanner import AudioTask


class BatchEndpointUnavailable(RuntimeError):
    """Raised when the server does not expose the required batch endpoint."""


class BatchTranscriptionClient:
    def __init__(self, server_url: str, timeout_seconds: int = 1800) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def transcribe_batch(
        self,
        tasks: list[AudioTask],
        route: str,
        language: str,
        timestamps: bool = True,
        speaker_diarization: bool = True,
    ) -> dict[str, Any]:
        url = f"{self.server_url}/asr/batch"
        handles = []
        try:
            files = []
            for task in tasks:
                handle = task.source_path.open("rb")
                handles.append(handle)
                files.append(("files", (task.relative_path.as_posix(), handle, "application/octet-stream")))
            data = {
                "model": route,
                "language": "" if language == "auto" else language,
                "timestamps": str(timestamps).lower(),
                "speaker_diarization": str(speaker_diarization).lower(),
                "output_granularity": "sentence",
            }
            response = requests.post(url, files=files, data=data, timeout=self.timeout_seconds)
            if response.status_code == 404:
                raise BatchEndpointUnavailable(
                    f"server does not expose /asr/batch at {self.server_url}; restart the updated ASR service"
                )
            response.raise_for_status()
            payload = response.json()
        finally:
            for handle in handles:
                handle.close()

        if "results" not in payload or not isinstance(payload["results"], list):
            raise ValueError("batch response must contain a results list")
        return payload


def result_key(result: dict[str, Any]) -> str:
    return str(result.get("file_name") or result.get("filename") or result.get("file") or "")
