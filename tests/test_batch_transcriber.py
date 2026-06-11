import json
from pathlib import Path

import pytest

from batch_transcriber.config import BatchConfig
from batch_transcriber.formatter import postprocess_segments, render_markdown, render_srt
from batch_transcriber.runner import run
from batch_transcriber.scanner import AudioTask, discover_audio_files
from batch_transcriber.client import BatchEndpointUnavailable, BatchTranscriptionClient


def test_scanner_recurses_and_preserves_output_tree(tmp_path):
    input_dir = tmp_path / "音频文件"
    output_dir = tmp_path / "听译结果"
    audio = input_dir / "会议" / "2026" / "录音.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    (input_dir / ".hidden.wav").write_bytes(b"audio")
    (input_dir / "empty.mp3").write_bytes(b"")
    (input_dir / "note.txt").write_text("skip", encoding="utf-8")

    tasks = discover_audio_files(input_dir, output_dir, (".wav", ".mp3"), recursive=True)

    assert [task.relative_path.as_posix() for task in tasks] == ["会议/2026/录音.wav"]
    assert tasks[0].output_md == output_dir / "会议" / "2026" / "录音.md"
    assert tasks[0].output_srt == output_dir / "会议" / "2026" / "录音.srt"
    assert tasks[0].output_json == output_dir / "会议" / "2026" / "录音.json"


def test_scanner_skips_completed_outputs_unless_overwrite(tmp_path):
    input_dir = tmp_path / "音频文件"
    output_dir = tmp_path / "听译结果"
    audio = input_dir / "录音.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"audio")
    for suffix in ("md", "srt", "json"):
        output_dir.mkdir(exist_ok=True)
        (output_dir / f"录音.{suffix}").write_text("done", encoding="utf-8")

    assert discover_audio_files(input_dir, output_dir, (".wav",), overwrite=False) == []
    assert len(discover_audio_files(input_dir, output_dir, (".wav",), overwrite=True)) == 1


def test_formatters_render_sentence_timestamps_speaker_and_empty_vad(tmp_path):
    task = AudioTask(
        source_path=tmp_path / "音频文件" / "录音.wav",
        relative_path=Path("录音.wav"),
        output_md=tmp_path / "听译结果" / "录音.md",
        output_srt=tmp_path / "听译结果" / "录音.srt",
        output_json=tmp_path / "听译结果" / "录音.json",
    )
    result = {
        "duration": 5.0,
        "processing_time": 1.0,
        "segments": [{"start": 0.42, "end": 3.8, "speaker": "SPK0", "text": "A|B"}],
    }

    md = render_markdown(task, result, "fun-asr-nano-vllm", "auto")
    srt = render_srt(result)

    assert "00:00:00.420 --> 00:00:03.800" in md
    assert "A\\|B" in md
    assert "[SPK0] A|B" in srt
    assert "00:00:00,420 --> 00:00:03,800" in srt
    assert "服务端 VAD 未检测到有效语音" in render_markdown(task, {"segments": []}, "route", "auto")
    assert "00:00:00,000 --> 00:00:00,000" in render_srt({"segments": []})
    assert "服务端 VAD 未检测到有效语音" in render_markdown(
        task,
        {"segments": [{"start": 0, "end": 1, "speaker": "SPK0", "text": ""}]},
        "route",
        "auto",
    )


def test_postprocess_merges_short_segments_and_filler():
    result = {
        "segments": [
            {"start": 0, "end": 1, "speaker": "SPK0", "text": "啊"},
            {"start": 1, "end": 5, "speaker": "SPK0", "text": "我们先看这个定义。"},
            {"start": 5, "end": 8, "speaker": "SPK0", "text": "嗯。"},
            {"start": 8, "end": 17, "speaker": "SPK0", "text": "然后把它写成矩阵表示。"},
        ]
    }

    segments = postprocess_segments(result, min_segment_seconds=15, target_segment_seconds=30)

    assert len(segments) == 1
    assert segments[0]["start"] == 0
    assert segments[0]["end"] == 17
    assert "我们先看这个定义" in segments[0]["text"]
    assert "然后把它写成矩阵表示" in segments[0]["text"]


def test_postprocess_stabilizes_short_speaker_turns():
    result = {
        "segments": [
            {"start": 0, "end": 10, "speaker": "SPK0", "text": "第一段内容。"},
            {"start": 10, "end": 13, "speaker": "SPK1", "text": "短暂误分。"},
            {"start": 13, "end": 24, "speaker": "SPK0", "text": "继续主讲内容。"},
        ]
    }

    segments = postprocess_segments(result, min_segment_seconds=15, min_speaker_turn_seconds=8)

    assert len(segments) == 1
    assert segments[0]["speaker"] == "SPK0"
    assert "短暂误分" in segments[0]["text"]


def test_postprocess_splits_at_max_segment_seconds():
    result = {
        "segments": [
            {"start": 0, "end": 20, "speaker": "SPK0", "text": "第一句。"},
            {"start": 20, "end": 40, "speaker": "SPK0", "text": "第二句。"},
            {"start": 40, "end": 65, "speaker": "SPK0", "text": "第三句。"},
        ]
    }

    segments = postprocess_segments(result, min_segment_seconds=15, target_segment_seconds=30, max_segment_seconds=45)

    assert len(segments) == 2
    assert segments[0]["end"] == 40
    assert segments[1]["start"] == 40


def test_postprocess_does_not_merge_across_large_silence_gap():
    result = {
        "segments": [
            {"start": 0, "end": 1, "speaker": "SPK0", "text": "啊"},
            {"start": 60, "end": 80, "speaker": "SPK0", "text": "这里开始讲正文。"},
        ]
    }

    segments = postprocess_segments(result, min_segment_seconds=15, max_merge_gap_seconds=8)

    assert len(segments) == 1
    assert segments[0]["start"] == 60
    assert "这里开始讲正文" in segments[0]["text"]


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def transcribe_batch(self, tasks, route, language, timestamps=True, speaker_diarization=True):
        self.calls.append([task.relative_path.as_posix() for task in tasks])
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def test_runner_batches_writes_outputs_and_failed_jsonl(tmp_path):
    input_dir = tmp_path / "音频文件"
    output_dir = tmp_path / "听译结果"
    files = []
    for name in ("a.wav", "nested/b.wav", "c.wav"):
        path = input_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")
        files.append(path)

    client = FakeClient(
        [
            {
                "results": [
                    {
                        "file_name": "a.wav",
                        "status": "success",
                        "duration": 1,
                        "segments": [{"start": 0, "end": 1, "speaker": "SPK0", "text": "hello"}],
                    },
                    {"file_name": "c.wav", "status": "failed", "error": "decode failed"},
                ]
            },
            RuntimeError("HTTP 500: CUDA out of memory"),
        ]
    )
    config = BatchConfig(input_dir=input_dir, output_dir=output_dir, batch_size=2, retries=0)

    summary = run(config, client=client)

    assert summary.total == 3
    assert summary.success == 1
    assert summary.failed == 2
    assert (output_dir / "a.md").exists()
    assert (output_dir / "a.srt").exists()
    assert json.loads((output_dir / "a.json").read_text(encoding="utf-8"))["translate"] is False
    failed = (output_dir / "failed_files.jsonl").read_text(encoding="utf-8")
    assert "decode failed" in failed
    assert "CUDA out of memory" in failed
    assert client.calls == [["a.wav", "c.wav"], ["nested/b.wav"]]


def test_runner_retry_failed_only_processes_failed_list(tmp_path):
    input_dir = tmp_path / "音频文件"
    output_dir = tmp_path / "听译结果"
    for name in ("done.wav", "failed.wav", "other.wav"):
        path = input_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")

    for suffix in ("md", "srt", "json"):
        output_dir.mkdir(exist_ok=True)
        (output_dir / f"done.{suffix}").write_text("done", encoding="utf-8")

    failed_payload = {
        "file": str(input_dir / "failed.wav"),
        "output_md": str(output_dir / "failed.md"),
        "output_srt": str(output_dir / "failed.srt"),
        "output_json": str(output_dir / "failed.json"),
    }
    (output_dir / "failed_files.jsonl").write_text(json.dumps(failed_payload, ensure_ascii=False) + "\n", encoding="utf-8")

    client = FakeClient(
        [
            {
                "results": [
                    {
                        "file_name": "failed.wav",
                        "status": "success",
                        "duration": 1,
                        "segments": [{"start": 0, "end": 1, "text": "retry ok"}],
                    }
                ]
            }
        ]
    )

    summary = run(BatchConfig(input_dir=input_dir, output_dir=output_dir, retry_failed=True), client=client)

    assert summary.total == 1
    assert summary.success == 1
    assert client.calls == [["failed.wav"]]
    assert (output_dir / "failed.md").exists()
    assert not (output_dir / "other.md").exists()


def test_client_reports_missing_batch_endpoint(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")
    task = AudioTask(
        source_path=audio,
        relative_path=Path("a.wav"),
        output_md=tmp_path / "a.md",
        output_srt=tmp_path / "a.srt",
        output_json=tmp_path / "a.json",
    )

    class Response:
        status_code = 404
        text = "not found"

        def raise_for_status(self):
            import requests

            raise requests.HTTPError("404 Client Error", response=self)

    monkeypatch.setattr("batch_transcriber.client.requests.post", lambda *args, **kwargs: Response())

    with pytest.raises(BatchEndpointUnavailable, match="/asr/batch"):
        BatchTranscriptionClient("http://127.0.0.1:8000").transcribe_batch([task], "fun-asr-nano-vllm", "auto")


def test_config_rejects_translation():
    with pytest.raises(ValueError, match="translation"):
        BatchConfig(translate=True).validate()
