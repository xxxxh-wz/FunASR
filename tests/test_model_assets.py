from pathlib import Path

from batch_transcriber.download_models import main
from batch_transcriber.model_assets import download_model, resolve_model_id


def test_resolve_model_aliases():
    assert resolve_model_id("fun-asr-nano-vllm") == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert resolve_model_id("qwen3-asr") == "Qwen/Qwen3-ASR-1.7B"
    assert resolve_model_id("fsmn-vad") == "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    assert resolve_model_id("custom/model") == "custom/model"


def test_download_model_uses_modelscope_cache_dir(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(model_id, cache_dir):
        calls.append((model_id, cache_dir))
        return str(Path(cache_dir) / model_id.replace("/", "__"))

    monkeypatch.setattr("batch_transcriber.model_assets.snapshot_download_model", fake_snapshot_download)

    asset = download_model("fun-asr-nano", cache_dir=tmp_path)

    assert calls == [("FunAudioLLM/Fun-ASR-Nano-2512", str(tmp_path))]
    assert asset.local_path == tmp_path / "FunAudioLLM__Fun-ASR-Nano-2512"


def test_download_models_cli_lists_aliases(capsys):
    assert main(["--list-aliases"]) == 0
    out = capsys.readouterr().out
    assert "fun-asr-nano" in out
    assert "qwen3-asr" in out
