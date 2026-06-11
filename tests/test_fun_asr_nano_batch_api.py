import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_server_module():
    path = Path("examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py")
    spec = importlib.util.spec_from_file_location("serve_vllm_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeUpload:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def _wav_upload(filename="a.wav"):
    import io

    data = io.BytesIO()
    sf.write(data, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    return FakeUpload(filename, data.getvalue())


def _json_response_payload(response):
    return json.loads(response.body.decode("utf-8"))


def test_asr_batch_returns_per_file_results(monkeypatch, tmp_path):
    module = _load_server_module()
    monkeypatch.setattr(module, "load_engine", lambda args: None)

    def fake_process_audio(audio_data, sr=16000, language=None, hotwords=None, use_vad=True, use_spk=False, use_timestamp=True):
        return {
            "text": f"{language or 'auto'} text",
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPK0" if use_spk else "", "text": "hello"}],
        }

    monkeypatch.setattr(module, "process_audio", fake_process_audio)

    response = module.asyncio.run(
        module.asr_batch_endpoint(
            files=[_wav_upload("dir/a.wav"), _wav_upload("b.wav")],
            model="fun-asr-nano-vllm",
            language="中文",
            hotwords="",
            speaker_diarization=True,
            timestamps=True,
            output_granularity="sentence",
        )
    )

    assert response.status_code == 200
    payload = _json_response_payload(response)
    assert payload["model"] == "fun-asr-nano-vllm"
    assert payload["batch_size"] == 2
    assert [item["status"] for item in payload["results"]] == ["success", "success"]
    assert payload["results"][0]["file_name"] == "dir/a.wav"
    assert payload["results"][0]["segments"][0]["speaker"] == "SPK0"


def test_asr_batch_records_decode_failure(monkeypatch, tmp_path):
    module = _load_server_module()
    monkeypatch.setattr(module, "load_engine", lambda args: None)
    monkeypatch.setattr(module, "read_audio_upload", lambda content, filename: (_ for _ in ()).throw(RuntimeError("decode failed")))

    response = module.asyncio.run(
        module.asr_batch_endpoint(
            files=[FakeUpload("bad.wav", b"bad")],
            model="fun-asr-nano-vllm",
            language=None,
            hotwords="",
            speaker_diarization=True,
            timestamps=True,
            output_granularity="sentence",
        )
    )

    assert response.status_code == 200
    result = _json_response_payload(response)["results"][0]
    assert result["status"] == "failed"
    assert "decode failed" in result["error"]


def test_asr_batch_routes_qwen3_asr_to_non_vllm_processor(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module, "load_engine", lambda args: None)
    calls = {}

    def fake_process_audio_qwen3(
        audio_data,
        sr=16000,
        language=None,
        hotwords=None,
        hotword_prompt_template=None,
        use_vad=True,
        use_spk=False,
        use_timestamp=True,
    ):
        calls["language"] = language
        calls["use_spk"] = use_spk
        calls["use_timestamp"] = use_timestamp
        return {
            "text": "qwen text",
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPK0" if use_spk else "", "text": "qwen text"}],
        }

    monkeypatch.setattr(module, "process_audio_qwen3", fake_process_audio_qwen3)

    response = module.asyncio.run(
        module.asr_batch_endpoint(
            files=[_wav_upload("a.wav")],
            model="qwen3-asr",
            language="中文",
            hotwords="",
            speaker_diarization=True,
            timestamps=True,
            output_granularity="sentence",
        )
    )

    assert response.status_code == 200
    payload = _json_response_payload(response)
    assert payload["model"] == "qwen3-asr"
    assert payload["results"][0]["text"] == "qwen text"
    assert payload["results"][0]["model_path"] == "Qwen/Qwen3-ASR-1.7B"
    assert calls == {"language": "中文", "use_spk": True, "use_timestamp": True}


def test_qwen3_asr_uses_templated_hotword_context(monkeypatch):
    module = _load_server_module()
    module._args = type("Args", (), {})()
    calls = {}

    class FakeModel:
        def generate(self, input, **kwargs):
            calls["context"] = kwargs.get("context")
            return [{"text": "qwen text"}]

    monkeypatch.setattr(
        module,
        "prepare_qwen_segments",
        lambda audio_data, sr=16000, use_vad=True: (
            np.zeros(16000, dtype=np.float32),
            16000,
            [np.zeros(16000, dtype=np.float32)],
            [(0, 1000)],
        ),
    )
    monkeypatch.setattr(module, "load_qwen_model", lambda: FakeModel())

    module.process_audio_qwen3(
        np.zeros(16000, dtype=np.float32),
        sr=16000,
        hotwords=["线性映射", "矩阵表示"],
    )

    assert calls["context"] == "以下是本段音频可能出现的专有名词、课程术语或人名，请在转写时优先参考：线性映射、矩阵表示"


def test_asr_batch_routes_qwen3_asr_vllm_to_vllm_processor(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module, "load_engine", lambda args: None)
    calls = {}

    def fake_process_audio_qwen3_vllm(
        audio_data,
        sr=16000,
        language=None,
        hotwords=None,
        hotword_prompt_template=None,
        use_vad=True,
        use_spk=False,
        use_timestamp=True,
    ):
        calls["language"] = language
        calls["hotwords"] = hotwords
        calls["use_spk"] = use_spk
        calls["use_timestamp"] = use_timestamp
        return {
            "text": "qwen vllm text",
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "", "text": "qwen vllm text"}],
        }

    monkeypatch.setattr(module, "process_audio_qwen3_vllm", fake_process_audio_qwen3_vllm)

    response = module.asyncio.run(
        module.asr_batch_endpoint(
            files=[_wav_upload("a.wav")],
            model="qwen3-asr-vllm",
            language="中文",
            hotwords="线性代数,矩阵",
            speaker_diarization=False,
            timestamps=True,
            output_granularity="sentence",
        )
    )

    assert response.status_code == 200
    payload = _json_response_payload(response)
    assert payload["model"] == "qwen3-asr-vllm"
    assert payload["results"][0]["text"] == "qwen vllm text"
    assert payload["results"][0]["model_path"] == "Qwen/Qwen3-ASR-1.7B"
    assert calls == {
        "language": "中文",
        "hotwords": ["线性代数", "矩阵"],
        "use_spk": False,
        "use_timestamp": True,
    }
    assert payload["hotwords_applied"] is True
    assert payload["hotwords_count"] == 2
    assert payload["results"][0]["hotwords_applied"] is True
    assert payload["results"][0]["hotwords_count"] == 2


def test_qwen3_vllm_uses_templated_hotword_context(monkeypatch):
    module = _load_server_module()
    module._args = type("Args", (), {"qwen_forced_aligner": ""})()
    calls = {}

    @dataclass
    class FakeTranscription:
        text: str
        time_stamps: None = None

    class FakeModel:
        def transcribe(self, **kwargs):
            calls["context"] = kwargs.get("context")
            return [FakeTranscription("qwen text")]

    monkeypatch.setattr(
        module,
        "prepare_qwen_segments",
        lambda audio_data, sr=16000, use_vad=True: (
            np.zeros(16000, dtype=np.float32),
            16000,
            [np.zeros(16000, dtype=np.float32)],
            [(0, 1000)],
        ),
    )
    monkeypatch.setattr(module, "load_qwen_vllm_model", lambda: FakeModel())

    module.process_audio_qwen3_vllm(
        np.zeros(16000, dtype=np.float32),
        sr=16000,
        hotwords=["线性映射", "矩阵表示"],
    )

    assert calls["context"] == "以下是本段音频可能出现的专有名词、课程术语或人名，请在转写时优先参考：线性映射、矩阵表示"


def test_qwen3_vllm_forced_aligner_items_are_mapped_to_words(monkeypatch):
    module = _load_server_module()
    module._args = type("Args", (), {"qwen_forced_aligner": "/models/fa"})()

    @dataclass
    class FakeAlignItem:
        text: str
        start_time: float
        end_time: float

    class FakeAlignResult:
        def __iter__(self):
            return iter(
                [
                    FakeAlignItem("你", 0.01, 0.12),
                    FakeAlignItem("好", 0.13, 0.24),
                ]
            )

    @dataclass
    class FakeTranscription:
        text: str
        time_stamps: FakeAlignResult

    class FakeModel:
        def transcribe(self, **kwargs):
            assert kwargs["return_time_stamps"] is True
            return [FakeTranscription("你好", FakeAlignResult())]

    monkeypatch.setattr(
        module,
        "prepare_qwen_segments",
        lambda audio_data, sr=16000, use_vad=True: (
            np.zeros(16000, dtype=np.float32),
            16000,
            [np.zeros(16000, dtype=np.float32)],
            [(1000, 2000)],
        ),
    )
    monkeypatch.setattr(module, "load_qwen_vllm_model", lambda: FakeModel())

    result = module.process_audio_qwen3_vllm(
        np.zeros(16000, dtype=np.float32),
        sr=16000,
        language="中文",
        use_timestamp=True,
    )

    assert result["segments"] == [
        {
            "text": "你好",
            "start": 1.0,
            "end": 2.0,
            "words": [
                {"word": "你", "start": 1.01, "end": 1.12},
                {"word": "好", "start": 1.13, "end": 1.24},
            ],
        }
    ]


def test_read_audio_upload_falls_back_to_librosa(monkeypatch, tmp_path):
    module = _load_server_module()
    calls = {}

    def fail_soundfile(content):
        raise RuntimeError("unsupported format")

    def fake_load(path, sr=None, mono=False):
        calls["path"] = path
        calls["sr"] = sr
        calls["mono"] = mono
        return np.zeros(16000, dtype=np.float32), 16000

    monkeypatch.setattr(module.sf, "read", fail_soundfile)
    monkeypatch.setattr("librosa.load", fake_load)

    audio, sr = module.read_audio_upload(b"fake m4a", "meeting.m4a")

    assert sr == 16000
    assert audio.shape == (16000,)
    assert calls["path"].endswith(".m4a")
    assert calls["sr"] is None
    assert calls["mono"] is False


def test_asr_batch_rejects_too_many_files(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module, "load_engine", lambda args: None)
    module._args = type("Args", (), {"max_batch_size": 1})()

    response = module.asyncio.run(
        module.asr_batch_endpoint(
            files=[_wav_upload("a.wav"), _wav_upload("b.wav")],
            model="fun-asr-nano-vllm",
            language=None,
            hotwords="",
            speaker_diarization=True,
            timestamps=True,
            output_granularity="sentence",
        )
    )

    assert response.status_code == 400
    assert "max_batch_size" in _json_response_payload(response)["error"]


def test_asr_batch_rejects_unsupported_route(monkeypatch):
    module = _load_server_module()
    monkeypatch.setattr(module, "load_engine", lambda args: None)

    response = module.asyncio.run(
        module.asr_batch_endpoint(
            files=[_wav_upload("a.wav")],
            model="unknown-asr",
            language=None,
            hotwords="",
            speaker_diarization=True,
            timestamps=True,
            output_granularity="sentence",
        )
    )

    assert response.status_code == 400
    payload = _json_response_payload(response)
    assert "unsupported model route" in payload["error"]
    assert "fun-asr-nano-vllm" in payload["supported_routes"]
    assert "qwen3-asr" in payload["supported_routes"]
    assert "qwen3-asr-vllm" in payload["supported_routes"]
