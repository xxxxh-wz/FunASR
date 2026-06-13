#!/usr/bin/env python3
"""Fun-ASR-Nano vLLM Inference Server.

Unified server with three interfaces:
- HTTP REST: POST /asr (file upload)
- WebSocket: ws://host:port/ws (streaming audio)
- OpenAI API: POST /v1/audio/transcriptions (Whisper-compatible)

All endpoints share the same vLLM engine + dynamic VAD + SPK + timestamps.

Usage:
    CUDA_VISIBLE_DEVICES=0 python serve_vllm.py --port 8000
    CUDA_VISIBLE_DEVICES=0 python serve_vllm.py --port 8000 --model FunAudioLLM/Fun-ASR-Nano-2512
"""

import asyncio
import argparse
import io
import json
import logging
import os
import re
import time
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
import torch
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def truncate_repetition(text, min_repeat_len=3, max_repeats=3):
    """Detect and truncate repetitive patterns in ASR output."""
    if not text or len(text) < 20:
        return text
    n = len(text)
    for length in range(min_repeat_len, min(n // max_repeats, 30)):
        for start in range(n - length * max_repeats):
            chunk = text[start:start + length]
            if text[start:start + length * max_repeats] == chunk * max_repeats:
                return text[:start + length]
    return text



try:
    from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response
    import uvicorn
except ImportError:
    raise ImportError("pip install fastapi uvicorn python-multipart")


# ============================================================
# Global state
# ============================================================
_engine = None
_qwen_model = None
_qwen_vllm_model = None
_vad_model = None
_spk_model = None
_args = None
_model_load_lock = threading.Lock()
_model_init_lock = threading.Lock()
_engine_inference_lock = threading.Lock()
_qwen_inference_lock = threading.Lock()
_qwen_vllm_inference_lock = threading.Lock()
_vad_inference_lock = threading.Lock()
_model_load_state = {
    "fun_asr_nano_vllm": {"status": "unloaded", "started_at": None, "ended_at": None, "error": None},
    "qwen3_asr": {"status": "unloaded", "started_at": None, "ended_at": None, "error": None},
    "qwen3_asr_vllm": {"status": "unloaded", "started_at": None, "ended_at": None, "error": None},
}
NANO_BATCH_ROUTES = {"fun-asr-nano-vllm", "fun-asr-nano", "FunAudioLLM/Fun-ASR-Nano-2512"}
QWEN3_BATCH_ROUTES = {"qwen3-asr", "qwen3-asr-1.7b", "Qwen/Qwen3-ASR-1.7B"}
QWEN3_VLLM_BATCH_ROUTES = {"qwen3-asr-vllm", "qwen3-asr-1.7b-vllm"}
SUPPORTED_BATCH_ROUTES = NANO_BATCH_ROUTES | QWEN3_BATCH_ROUTES | QWEN3_VLLM_BATCH_ROUTES
DEFAULT_HOTWORD_PROMPT_TEMPLATE = "以下是本段音频可能出现的专有名词、课程术语或人名，请在转写时优先参考：{hotwords}"


def log_stage(stage, started_at, **fields):
    elapsed = time.perf_counter() - started_at
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    if details:
        logger.info("ASR stage=%s elapsed=%.3fs %s", stage, elapsed, details)
    else:
        logger.info("ASR stage=%s elapsed=%.3fs", stage, elapsed)
    return time.perf_counter()


def parse_hotwords(raw_hotwords):
    if not raw_hotwords:
        return None
    parts = re.split(r"[,，、\n\r]+", str(raw_hotwords))
    hotwords = []
    seen = set()
    for part in parts:
        item = part.strip()
        if not item or item.startswith("#") or item in seen:
            continue
        seen.add(item)
        hotwords.append(item)
    return hotwords or None


def build_hotword_context(hotwords=None, template=None):
    if not hotwords:
        return None
    text = "、".join(str(word).strip() for word in hotwords if str(word).strip())
    if not text:
        return None
    prompt_template = template or DEFAULT_HOTWORD_PROMPT_TEMPLATE
    return prompt_template.format(hotwords=text)


def route_state_key(model):
    if model in QWEN3_VLLM_BATCH_ROUTES:
        return "qwen3_asr_vllm"
    if model in QWEN3_BATCH_ROUTES:
        return "qwen3_asr"
    if model in NANO_BATCH_ROUTES:
        return "fun_asr_nano_vllm"
    return None


def set_model_load_state(key, status, *, error=None):
    now = time.time()
    with _model_load_lock:
        state = _model_load_state.setdefault(
            key,
            {"status": "unloaded", "started_at": None, "ended_at": None, "error": None},
        )
        if status == "loading":
            state.update({"status": status, "started_at": now, "ended_at": None, "error": None})
        elif status in {"loaded", "failed"}:
            state.update({"status": status, "ended_at": now, "error": error})
            if state.get("started_at") is None:
                state["started_at"] = now
        else:
            state.update({"status": status, "error": error})


def model_load_status_payload():
    with _model_load_lock:
        status = {
            key: {
                "status": value.get("status"),
                "started_at": value.get("started_at"),
                "ended_at": value.get("ended_at"),
                "elapsed_seconds": (
                    round((value.get("ended_at") or time.time()) - value["started_at"], 3)
                    if value.get("started_at") else None
                ),
                "error": value.get("error"),
            }
            for key, value in _model_load_state.items()
        }

    if _engine is not None:
        status["fun_asr_nano_vllm"]["status"] = "loaded"
    if _qwen_model is not None:
        status["qwen3_asr"]["status"] = "loaded"
    if _qwen_vllm_model is not None:
        status["qwen3_asr_vllm"]["status"] = "loaded"
    return status


def spk_model_kwargs(spk_model, device):
    kwargs = {"model": spk_model, "device": device, "disable_update": True}
    if spk_model and os.path.exists(spk_model):
        kwargs["model"] = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
        kwargs["model_path"] = spk_model
    return kwargs


def health_payload():
    load_status = model_load_status_payload()
    loading_models = [key for key, value in load_status.items() if value.get("status") == "loading"]
    return {
        "status": "loading" if loading_models else "ok",
        "loading_model": loading_models[0] if loading_models else None,
        "supported_routes": sorted(SUPPORTED_BATCH_ROUTES),
        "models_loaded": {
            "fun_asr_nano_vllm": _engine is not None,
            "qwen3_asr": _qwen_model is not None,
            "qwen3_asr_vllm": _qwen_vllm_model is not None,
            "vad": _vad_model is not None,
            "spk": _spk_model is not None,
        },
        "model_paths": {
            "fun_asr_nano_vllm": getattr(_args, "model", None) if _args is not None else None,
            "qwen3_asr": getattr(_args, "qwen_model", None) if _args is not None else None,
            "vad": getattr(_args, "vad_model", None) if _args is not None else None,
            "spk": getattr(_args, "spk_model", None) if _args is not None else None,
        },
        "model_load_status": load_status,
    }


def load_engine(args):
    global _engine, _vad_model, _spk_model, _args
    _args = args
    if _engine is None:
        if args is None:
            raise RuntimeError("server args are not initialized")
        from funasr import AutoModel
        from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

        set_model_load_state("fun_asr_nano_vllm", "loading")
        try:
            logger.info(f"Loading vLLM engine: {args.model}")
            _engine = FunASRNanoVLLM.from_pretrained(
                model=args.model, hub=args.hub, device=args.device, dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            logger.info(f"Loading VAD: {args.vad_model}")
            _vad_model = AutoModel(model=args.vad_model, device=args.device, disable_update=True)
            if args.spk_model:
                logger.info(f"Loading SPK: {args.spk_model}")
                _spk_model = AutoModel(**spk_model_kwargs(args.spk_model, args.device))
            else:
                logger.info("SPK disabled")
            set_model_load_state("fun_asr_nano_vllm", "loaded")
            logger.info("All models ready!")
        except Exception as exc:
            set_model_load_state("fun_asr_nano_vllm", "failed", error=str(exc))
            raise


def load_qwen_model():
    global _qwen_model
    if _qwen_model is None:
        if _args is None:
            raise RuntimeError("server args are not initialized")
        from funasr import AutoModel

        set_model_load_state("qwen3_asr", "loading")
        try:
            logger.info(f"Loading Qwen3-ASR model: {_args.qwen_model}")
            qwen_kwargs = {
                "model": _args.qwen_model,
                "hub": _args.hub,
                "device": _args.device,
                "dtype": _args.dtype,
                "disable_update": True,
            }
            if os.path.exists(_args.qwen_model):
                qwen_kwargs["model"] = "Qwen/Qwen3-ASR-1.7B"
                qwen_kwargs["model_path"] = _args.qwen_model
            _qwen_model = AutoModel(
                **qwen_kwargs,
            )
            set_model_load_state("qwen3_asr", "loaded")
            logger.info("Qwen3-ASR model ready!")
        except Exception as exc:
            set_model_load_state("qwen3_asr", "failed", error=str(exc))
            raise
    return _qwen_model


def load_qwen_vllm_model():
    global _qwen_vllm_model
    if _qwen_vllm_model is None:
        if _args is None:
            raise RuntimeError("server args are not initialized")
        from qwen_asr import Qwen3ASRModel

        set_model_load_state("qwen3_asr_vllm", "loading")
        try:
            logger.info(f"Loading Qwen3-ASR vLLM model: {_args.qwen_model}")
            forced_aligner = getattr(_args, "qwen_forced_aligner", "") or None
            forced_aligner_kwargs = None
            if forced_aligner:
                forced_aligner_kwargs = {
                    "dtype": getattr(torch, str(_args.dtype).replace("bf16", "bfloat16"), torch.bfloat16),
                    "device_map": _args.device,
                }
            qwen_kwargs = {
                "model": _args.qwen_model,
                "gpu_memory_utilization": getattr(_args, "qwen_gpu_memory_utilization", None) or _args.gpu_memory_utilization,
                "max_inference_batch_size": getattr(_args, "qwen_max_inference_batch_size", 128),
                "max_new_tokens": getattr(_args, "qwen_max_new_tokens", 4096),
            }
            qwen_max_model_len = getattr(_args, "qwen_max_model_len", None)
            if qwen_max_model_len:
                qwen_kwargs["max_model_len"] = qwen_max_model_len
            if forced_aligner:
                qwen_kwargs["forced_aligner"] = forced_aligner
                qwen_kwargs["forced_aligner_kwargs"] = forced_aligner_kwargs
            _qwen_vllm_model = Qwen3ASRModel.LLM(**qwen_kwargs)
            set_model_load_state("qwen3_asr_vllm", "loaded")
            logger.info("Qwen3-ASR vLLM model ready!")
        except Exception as exc:
            set_model_load_state("qwen3_asr_vllm", "failed", error=str(exc))
            raise
    return _qwen_vllm_model


def normalize_qwen_language(language):
    if not language or language == "auto":
        return None
    language_map = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "cn": "Chinese",
        "中文": "Chinese",
        "汉语": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
        "英文": "English",
    }
    return language_map.get(str(language).strip().lower(), language)


def audio_shape_info(audio_data):
    shape = getattr(audio_data, "shape", None)
    if shape is None:
        return None, None
    channels = shape[1] if len(shape) > 1 else 1
    return "x".join(str(dim) for dim in shape), channels


def normalize_audio_for_asr(audio_data, sr=16000, *, stage_prefix="audio"):
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    audio_data = audio_data.astype(np.float32)
    if sr != 16000:
        resample_t0 = time.perf_counter()
        original_sr = sr
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
        log_stage(
            f"{stage_prefix}_resample",
            resample_t0,
            from_sr=original_sr,
            to_sr=sr,
            samples=len(audio_data),
        )
    return audio_data, sr


def run_vad(audio_data, sr, *, use_vad=True, stage_prefix="audio"):
    if use_vad and len(audio_data) > sr * 1:
        logger.info("ASR stage=%s_vad_wait duration=%.3f", stage_prefix, len(audio_data) / sr)
        with _vad_inference_lock:
            vad_t0 = time.perf_counter()
            vad_res = _vad_model.generate(input=audio_data, fs=sr)
        segments = vad_res[0]["value"]
        log_stage(f"{stage_prefix}_vad_infer", vad_t0, segments=len(segments or []))
    else:
        segments = [[0, int(len(audio_data) * 1000 / sr)]]
    return segments


def prepare_qwen_segments(audio_data, sr=16000, use_vad=True, stage_prefix="qwen"):
    audio_data, sr = normalize_audio_for_asr(audio_data, sr, stage_prefix=stage_prefix)

    segments = run_vad(audio_data, sr, use_vad=use_vad, stage_prefix=stage_prefix)

    seg_audios = []
    seg_times = []
    for seg in segments or []:
        s0 = int(seg[0] * sr / 1000)
        s1 = int(seg[1] * sr / 1000)
        seg_audio = audio_data[s0:s1]
        if len(seg_audio) > sr * 0.3:
            seg_audios.append(seg_audio)
            seg_times.append((seg[0], seg[1]))
    return audio_data, sr, seg_audios, seg_times


def parse_qwen_time_stamp_item(ts):
    """Normalize qwen-asr forced aligner timestamp items."""
    if isinstance(ts, dict):
        word = ts.get("text") or ts.get("word") or ts.get("token") or ""
        ts_start = ts.get("start") or ts.get("start_time") or 0
        ts_end = ts.get("end") or ts.get("end_time") or 0
    elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
        word = ts[2] if len(ts) > 2 else ""
        ts_start, ts_end = ts[0], ts[1]
    elif all(hasattr(ts, name) for name in ("start_time", "end_time")):
        word = getattr(ts, "text", "") or getattr(ts, "word", "") or getattr(ts, "token", "")
        ts_start = getattr(ts, "start_time")
        ts_end = getattr(ts, "end_time")
    else:
        return None
    return {"word": str(word), "start": float(ts_start), "end": float(ts_end)}


def process_audio(audio_data, sr=16000, language=None, hotwords=None, 
                  use_vad=True, use_spk=False, use_timestamp=True):
    """Core processing: VAD segment → vLLM ASR → timestamps → SPK."""
    total_t0 = time.perf_counter()
    stage_t0 = time.perf_counter()
    original_duration = len(audio_data) / sr if sr else 0
    audio_data, sr = normalize_audio_for_asr(audio_data, sr, stage_prefix="nano")

    # VAD segmentation
    segments = run_vad(audio_data, sr, use_vad=use_vad, stage_prefix="nano")
    stage_t0 = log_stage("nano_vad", stage_t0, duration=round(original_duration, 3), segments=len(segments or []))

    if not segments:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    # Extract segment audio
    seg_audios = []
    seg_times = []
    for seg in segments:
        s0 = int(seg[0] * sr / 1000)
        s1 = int(seg[1] * sr / 1000)
        seg_audio = audio_data[s0:s1]
        if len(seg_audio) > sr * 0.3:
            seg_audios.append(seg_audio)
            seg_times.append((seg[0], seg[1]))

    if not seg_audios:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    # vLLM batch ASR
    gen_kwargs = {"max_new_tokens": 500}
    if language:
        gen_kwargs["language"] = language
    if hotwords:
        gen_kwargs["hotwords"] = hotwords

    logger.info("ASR stage=nano_vllm_wait segments=%s", len(seg_audios))
    with _engine_inference_lock:
        infer_t0 = time.perf_counter()
        results = _engine.generate(inputs=seg_audios, **gen_kwargs)
    log_stage("nano_vllm", infer_t0, segments=len(seg_audios))

    # Build segments with timestamps
    output_segments = []
    full_text_parts = []

    for i, (r, (start_ms, end_ms)) in enumerate(zip(results, seg_times)):
        r["text"] = truncate_repetition(r["text"])
        seg_info = {
            "text": r["text"],
            "start": start_ms / 1000,
            "end": end_ms / 1000,
        }
        if use_timestamp and "timestamps" in r:
            # Offset timestamps by segment start
            offset = start_ms / 1000
            seg_info["words"] = [
                {"word": ts["token"], "start": ts["start_time"] + offset, "end": ts["end_time"] + offset}
                for ts in r["timestamps"]
            ]
        output_segments.append(seg_info)
        full_text_parts.append(r["text"])

    # SPK diarization
    if use_spk and _spk_model is not None:
        spk_t0 = time.perf_counter()
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend

        vad_segs = [[st, et, audio_data[int(st*sr):int(et*sr)]] 
                    for st, et in [(s["start"], s["end"]) for s in output_segments]]
        chunks = sv_chunk(vad_segs)
        if chunks:
            speech_list = [ch[2] for ch in chunks]
            spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
            labels = cluster(embs.cpu(), oracle_num=None)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            all_sorted = sorted(chunks, key=lambda x: x[0])
            sv_output = postprocess(all_sorted, None, labels, embs.cpu())
            sentences = [{"text": s["text"], "start": int(s["start"]*1000), "end": int(s["end"]*1000)} 
                        for s in output_segments]
            distribute_spk(sentences, sv_output)
            for i, s in enumerate(sentences):
                output_segments[i]["speaker"] = f"SPK{s.get('spk', 0)}"
        log_stage("nano_spk", spk_t0, chunks=len(chunks) if "chunks" in locals() else 0)
    log_stage("nano_total", total_t0, output_segments=len(output_segments))

    return {
        "text": " ".join(full_text_parts),
        "segments": output_segments,
        "duration": len(audio_data) / sr,
    }


def process_audio_qwen3(audio_data, sr=16000, language=None, hotwords=None, hotword_prompt_template=None,
                        use_vad=True, use_spk=False, use_timestamp=True):
    """Core Qwen3-ASR processing: VAD segment → Qwen3-ASR batch → optional SPK."""
    total_t0 = time.perf_counter()
    input_duration = len(audio_data) / sr if sr else 0
    stage_t0 = time.perf_counter()
    audio_data, sr, seg_audios, seg_times = prepare_qwen_segments(audio_data, sr, use_vad)
    log_stage("qwen_vad", stage_t0, duration=round(input_duration, 3), segments=len(seg_audios))

    if not seg_audios:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    model = load_qwen_model()
    gen_kwargs = {"language": normalize_qwen_language(language)}
    if hotwords:
        gen_kwargs["context"] = build_hotword_context(hotwords, hotword_prompt_template)
    gen_kwargs = {key: value for key, value in gen_kwargs.items() if value}
    if use_timestamp:
        gen_kwargs["output_timestamp"] = True

    logger.info("ASR stage=qwen_wait segments=%s", len(seg_audios))
    with _qwen_inference_lock:
        infer_t0 = time.perf_counter()
        results = model.generate(input=seg_audios, **gen_kwargs)
    log_stage("qwen_infer", infer_t0, segments=len(seg_audios))
    output_segments = []
    full_text_parts = []
    for r, (start_ms, end_ms) in zip(results, seg_times):
        text = truncate_repetition(str(r.get("text") or ""))
        seg_info = {
            "text": text,
            "start": start_ms / 1000,
            "end": end_ms / 1000,
        }
        if "timestamp" in r:
            offset = start_ms / 1000
            seg_info["words"] = [
                {"start": ts[0] / 1000 + offset, "end": ts[1] / 1000 + offset}
                for ts in r["timestamp"]
            ]
        output_segments.append(seg_info)
        full_text_parts.append(text)

    if use_spk and _spk_model is not None:
        spk_t0 = time.perf_counter()
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend

        vad_segs = [[st, et, audio_data[int(st*sr):int(et*sr)]]
                    for st, et in [(s["start"], s["end"]) for s in output_segments]]
        chunks = sv_chunk(vad_segs)
        if chunks:
            speech_list = [ch[2] for ch in chunks]
            spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
            labels = cluster(embs.cpu(), oracle_num=None)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            all_sorted = sorted(chunks, key=lambda x: x[0])
            sv_output = postprocess(all_sorted, None, labels, embs.cpu())
            sentences = [{"text": s["text"], "start": int(s["start"]*1000), "end": int(s["end"]*1000)}
                         for s in output_segments]
            distribute_spk(sentences, sv_output)
            for i, s in enumerate(sentences):
                output_segments[i]["speaker"] = f"SPK{s.get('spk', 0)}"
        log_stage("qwen_spk", spk_t0, chunks=len(chunks) if "chunks" in locals() else 0)
    log_stage("qwen_total", total_t0, output_segments=len(output_segments))

    return {
        "text": " ".join(full_text_parts),
        "segments": output_segments,
        "duration": len(audio_data) / sr,
    }


def process_audio_qwen3_vllm(audio_data, sr=16000, language=None, hotwords=None, hotword_prompt_template=None,
                             use_vad=True, use_spk=False, use_timestamp=True):
    """Core Qwen3-ASR vLLM processing: VAD segment → qwen-asr vLLM batch → optional SPK."""
    total_t0 = time.perf_counter()
    input_duration = len(audio_data) / sr if sr else 0
    stage_t0 = time.perf_counter()
    audio_data, sr, seg_audios, seg_times = prepare_qwen_segments(audio_data, sr, use_vad)
    log_stage("qwen_vllm_vad", stage_t0, duration=round(input_duration, 3), segments=len(seg_audios))

    if not seg_audios:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    model = load_qwen_vllm_model()
    qwen_language = normalize_qwen_language(language)
    context = build_hotword_context(hotwords, hotword_prompt_template)
    has_forced_aligner = bool(getattr(_args, "qwen_forced_aligner", "") or "")
    logger.info("ASR stage=qwen_vllm_wait segments=%s timestamps=%s", len(seg_audios), bool(use_timestamp and has_forced_aligner))
    with _qwen_vllm_inference_lock:
        infer_t0 = time.perf_counter()
        results = model.transcribe(
            audio=[(seg_audio, sr) for seg_audio in seg_audios],
            language=qwen_language,
            context=context,
            return_time_stamps=bool(use_timestamp and has_forced_aligner),
        )
    log_stage("qwen_vllm_infer", infer_t0, segments=len(seg_audios))

    output_segments = []
    full_text_parts = []
    for r, (start_ms, end_ms) in zip(results, seg_times):
        text = truncate_repetition(str(getattr(r, "text", "") or ""))
        seg_info = {
            "text": text,
            "start": start_ms / 1000,
            "end": end_ms / 1000,
        }
        time_stamps = getattr(r, "time_stamps", None)
        if use_timestamp and time_stamps:
            offset = start_ms / 1000
            words = []
            for ts in time_stamps:
                parsed = parse_qwen_time_stamp_item(ts)
                if parsed is None:
                    continue
                words.append(
                    {
                        "word": parsed["word"],
                        "start": parsed["start"] + offset,
                        "end": parsed["end"] + offset,
                    }
                )
            if words:
                seg_info["words"] = words
        output_segments.append(seg_info)
        full_text_parts.append(text)

    if use_spk and _spk_model is not None:
        spk_t0 = time.perf_counter()
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend

        vad_segs = [[st, et, audio_data[int(st*sr):int(et*sr)]]
                    for st, et in [(s["start"], s["end"]) for s in output_segments]]
        chunks = sv_chunk(vad_segs)
        if chunks:
            speech_list = [ch[2] for ch in chunks]
            spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
            labels = cluster(embs.cpu(), oracle_num=None)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            all_sorted = sorted(chunks, key=lambda x: x[0])
            sv_output = postprocess(all_sorted, None, labels, embs.cpu())
            sentences = [{"text": s["text"], "start": int(s["start"]*1000), "end": int(s["end"]*1000)}
                         for s in output_segments]
            distribute_spk(sentences, sv_output)
            for i, s in enumerate(sentences):
                output_segments[i]["speaker"] = f"SPK{s.get('spk', 0)}"
        log_stage("qwen_vllm_spk", spk_t0, chunks=len(chunks) if "chunks" in locals() else 0)
    log_stage("qwen_vllm_total", total_t0, output_segments=len(output_segments))

    return {
        "text": " ".join(full_text_parts),
        "segments": output_segments,
        "duration": len(audio_data) / sr,
    }


def read_audio_upload(content, filename="audio"):
    """Read uploaded audio bytes, falling back to librosa/ffmpeg-backed decoders."""
    try:
        return sf.read(io.BytesIO(content))
    except Exception:
        suffix = os.path.splitext(filename or "")[1] or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            import librosa

            audio_data, sr = librosa.load(tmp.name, sr=None, mono=False)
        if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
            audio_data = audio_data.T
        return audio_data, sr


def validate_batch_request(files, model):
    if model not in SUPPORTED_BATCH_ROUTES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"unsupported model route for this service: {model}",
                "supported_routes": sorted(SUPPORTED_BATCH_ROUTES),
            },
        )

    max_batch_size = getattr(_args, "max_batch_size", 8) if _args is not None else 8
    if len(files) > max_batch_size:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"batch size {len(files)} exceeds max_batch_size {max_batch_size}",
                "max_batch_size": max_batch_size,
            },
        )
    return None


def select_batch_processor(model):
    if model in QWEN3_VLLM_BATCH_ROUTES:
        return process_audio_qwen3_vllm
    if model in QWEN3_BATCH_ROUTES:
        return process_audio_qwen3
    return process_audio


def load_model_for_route(model):
    with _model_init_lock:
        if model in QWEN3_VLLM_BATCH_ROUTES:
            load_qwen_vllm_model()
        elif model in QWEN3_BATCH_ROUTES:
            load_qwen_model()
        elif model in NANO_BATCH_ROUTES:
            load_engine(_args)
        else:
            raise ValueError(f"unsupported model route for this service: {model}")
    return health_payload()["model_load_status"][route_state_key(model)]


def preload_models(raw_models):
    models = [item.strip() for item in re.split(r"[,，\s]+", raw_models or "") if item.strip()]
    for model in models:
        logger.info("Preloading model route: %s", model)
        load_model_for_route(model)


def model_path_for_route(model):
    if model in (QWEN3_BATCH_ROUTES | QWEN3_VLLM_BATCH_ROUTES):
        return getattr(_args, "qwen_model", "Qwen/Qwen3-ASR-1.7B")
    return getattr(_args, "model", "FunAudioLLM/Fun-ASR-Nano-2512")


async def process_batch_upload_file(
    file,
    *,
    model,
    language=None,
    hotwords=None,
    hotword_prompt_template=None,
    speaker_diarization=True,
    timestamps=True,
):
    item_t0 = time.perf_counter()
    file_name = file.filename or "unknown"
    hotwords_count = len(hotwords or [])
    try:
        logger.info("ASR file_start name=%s model=%s spk=%s timestamps=%s", file_name, model, speaker_diarization, timestamps)
        read_t0 = time.perf_counter()
        content = await file.read()
        log_stage("upload_read", read_t0, file=file_name, bytes=len(content))
        decode_t0 = time.perf_counter()
        audio_data, sr = read_audio_upload(content, file_name)
        duration = len(audio_data) / sr if sr else 0
        audio_shape, audio_channels = audio_shape_info(audio_data)
        log_stage(
            "audio_decode",
            decode_t0,
            file=file_name,
            sr=sr,
            duration=round(duration, 3),
            shape=audio_shape,
            channels=audio_channels,
        )
        processor = select_batch_processor(model)
        processor_kwargs = {
            "language": language or None,
            "hotwords": hotwords,
            "use_spk": speaker_diarization,
            "use_timestamp": timestamps,
        }
        if processor in (process_audio_qwen3, process_audio_qwen3_vllm):
            processor_kwargs["hotword_prompt_template"] = hotword_prompt_template
        result = await asyncio.to_thread(
            processor,
            audio_data,
            sr=sr,
            **processor_kwargs,
        )
        item_elapsed = time.perf_counter() - item_t0
        logger.info(
            "ASR file_done name=%s model=%s elapsed=%.3fs duration=%.3fs rtf=%.4f status=success",
            file_name,
            model,
            item_elapsed,
            result.get("duration", 0),
            item_elapsed / result["duration"] if result.get("duration", 0) > 0 else 0,
        )
        result.update(
            {
                "file_name": file_name,
                "status": "success",
                "model_path": model_path_for_route(model),
                "processing_time": round(item_elapsed, 3),
                "rtf": round(item_elapsed / result["duration"], 4) if result.get("duration", 0) > 0 else 0,
                "hotwords_applied": hotwords_count > 0,
                "hotwords_count": hotwords_count,
            }
        )
        return result
    except Exception as exc:
        logger.exception("Batch ASR failed for %s", file_name)
        return {
            "file_name": file_name,
            "status": "failed",
            "error": str(exc),
            "processing_time": round(time.perf_counter() - item_t0, 3),
            "hotwords_applied": hotwords_count > 0,
            "hotwords_count": hotwords_count,
        }


def parse_output_formats(raw_formats):
    formats = [item.strip().lower() for item in re.split(r"[,，\s]+", raw_formats or "") if item.strip()]
    formats = formats or ["md", "srt", "json"]
    supported = {"md", "srt", "json"}
    unsupported = [item for item in formats if item not in supported]
    if unsupported:
        raise ValueError(f"unsupported output format: {', '.join(unsupported)}")
    return tuple(dict.fromkeys(formats))


def safe_zip_stem(file_name):
    raw_name = str(file_name or "audio").replace("\\", "/")
    parts = [part for part in PurePosixPath(raw_name).parts if part not in ("", ".", "..", "/")]
    if not parts:
        parts = ["audio"]
    return str(PurePosixPath(*parts).with_suffix(""))


def subtitle_error_json(result):
    return json.dumps(result, ensure_ascii=False, indent=2)


def subtitle_files_for_result(result, *, model, language, formats):
    from batch_transcriber.config import BatchConfig
    from batch_transcriber.formatter import render_json, render_markdown, render_srt
    from batch_transcriber.scanner import AudioTask

    stem = safe_zip_stem(result.get("file_name"))
    if result.get("status") != "success":
        return {f"{stem}.error.json": subtitle_error_json(result)}

    source_path = PurePosixPath(f"{stem}.audio")
    task = AudioTask(
        source_path=Path(source_path.as_posix()),
        relative_path=Path(source_path.as_posix()),
        output_md=Path(PurePosixPath(f"{stem}.md").as_posix()),
        output_srt=Path(PurePosixPath(f"{stem}.srt").as_posix()),
        output_json=Path(PurePosixPath(f"{stem}.json").as_posix()),
    )
    config = BatchConfig(route=model, language=language or "auto")
    outputs = {}
    if "md" in formats:
        outputs[f"{stem}.md"] = render_markdown(task, result, model, language or "auto", config=config)
    if "srt" in formats:
        outputs[f"{stem}.srt"] = render_srt(result, config=config)
    if "json" in formats:
        outputs[f"{stem}.json"] = render_json(task, result, model, language or "auto", config=config)
    return outputs


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Fun-ASR-Nano vLLM Server", version="1.0")


@app.on_event("startup")
async def startup():
    load_engine(_args)


@app.get("/healthz")
async def healthz():
    return health_payload()


@app.post("/models/{route:path}/load")
async def load_model_endpoint(route: str, background: bool = True):
    """Explicitly load a model route before the first transcription request."""
    if route not in SUPPORTED_BATCH_ROUTES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"unsupported model route for this service: {route}",
                "supported_routes": sorted(SUPPORTED_BATCH_ROUTES),
            },
        )

    state_key = route_state_key(route)
    current_status = model_load_status_payload()[state_key]
    if current_status.get("status") == "loaded":
        return JSONResponse(content={"route": route, **current_status})
    if current_status.get("status") == "loading":
        return JSONResponse(status_code=202, content={"route": route, **current_status})

    if background:
        set_model_load_state(state_key, "loading")
        asyncio.create_task(asyncio.to_thread(load_model_for_route, route))
        return JSONResponse(
            status_code=202,
            content={"route": route, "status": "loading", "detail": "model loading started"},
        )

    try:
        loaded_status = await asyncio.to_thread(load_model_for_route, route)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"route": route, "status": "failed", "error": str(exc)})
    return JSONResponse(content={"route": route, **loaded_status})


# --- HTTP REST: POST /asr ---
@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
    """ASR with file upload. Returns text + segments + timestamps + speaker."""
    content = await file.read()
    audio_data, sr = read_audio_upload(content, file.filename)

    hw_list = parse_hotwords(hotwords)

    t0 = time.perf_counter()
    result = process_audio(audio_data, sr=sr, language=language, 
                          hotwords=hw_list, use_spk=spk, use_timestamp=timestamp)
    t1 = time.perf_counter()

    result["processing_time"] = round(t1 - t0, 3)
    result["rtf"] = round((t1 - t0) / result["duration"], 4) if result["duration"] > 0 else 0
    return JSONResponse(content=result)


@app.post("/asr/batch")
async def asr_batch_endpoint(
    files: list[UploadFile] = File(...),
    model: str = Form(default="fun-asr-nano-vllm"),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    hotword_prompt_template: str = Form(default=""),
    speaker_diarization: bool = Form(default=True),
    timestamps: bool = Form(default=True),
    output_granularity: str = Form(default="sentence"),
):
    """Batch ASR with multiple file uploads in one request."""
    validation_error = validate_batch_request(files, model)
    if validation_error is not None:
        return validation_error

    hw_list = parse_hotwords(hotwords)
    hotwords_count = len(hw_list or [])
    hotword_template = hotword_prompt_template or None
    results = []
    batch_t0 = time.perf_counter()

    for file in files:
        results.append(
            await process_batch_upload_file(
                file,
                model=model,
                language=language,
                hotwords=hw_list,
                hotword_prompt_template=hotword_template,
                speaker_diarization=speaker_diarization,
                timestamps=timestamps,
            )
        )

    return JSONResponse(
        content={
            "model": model,
            "batch_size": len(files),
            "output_granularity": output_granularity,
            "hotwords_applied": hotwords_count > 0,
            "hotwords_count": hotwords_count,
            "processing_time": round(time.perf_counter() - batch_t0, 3),
            "results": results,
        }
    )


@app.post("/asr/subtitles")
async def asr_subtitles_endpoint(
    files: list[UploadFile] = File(...),
    model: str = Form(default="qwen3-asr-vllm"),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    hotword_prompt_template: str = Form(default=""),
    speaker_diarization: bool = Form(default=True),
    timestamps: bool = Form(default=True),
    output_granularity: str = Form(default="sentence"),
    output_formats: str = Form(default="md,srt,json"),
):
    """Batch ASR and return generated subtitle/transcript files as a zip archive."""
    validation_error = validate_batch_request(files, model)
    if validation_error is not None:
        return validation_error

    try:
        formats = parse_output_formats(output_formats)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc), "supported_formats": ["md", "srt", "json"]})

    hw_list = parse_hotwords(hotwords)
    hotword_template = hotword_prompt_template or None
    buffer = io.BytesIO()
    request_t0 = time.perf_counter()
    logger.info(
        "ASR subtitles_start model=%s files=%s formats=%s spk=%s timestamps=%s hotwords=%s",
        model,
        len(files),
        ",".join(formats),
        speaker_diarization,
        timestamps,
        len(hw_list or []),
    )
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            result = await process_batch_upload_file(
                file,
                model=model,
                language=language,
                hotwords=hw_list,
                hotword_prompt_template=hotword_template,
                speaker_diarization=speaker_diarization,
                timestamps=timestamps,
            )
            for archive_name, content in subtitle_files_for_result(
                result,
                model=model,
                language=language,
                formats=formats,
            ).items():
                zip_t0 = time.perf_counter()
                archive.writestr(archive_name, content)
                log_stage("zip_write", zip_t0, file=archive_name, bytes=len(content.encode("utf-8")))

    buffer.seek(0)
    logger.info("ASR subtitles_done model=%s elapsed=%.3fs zip_bytes=%s", model, time.perf_counter() - request_t0, len(buffer.getvalue()))
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="subtitles.zip"'},
    )


# --- OpenAI API: POST /v1/audio/transcriptions ---
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="fun-asr-nano"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: str = Form(default="word"),
    spk: bool = Form(default=False),
):
    """OpenAI Whisper-compatible transcription API (extended with spk support)."""
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))

    use_ts = "word" in timestamp_granularities or "segment" in timestamp_granularities
    result = process_audio(audio_data, sr=sr, language=language, use_spk=spk, use_timestamp=use_ts)

    if response_format == "text":
        return JSONResponse(content=result["text"])
    elif response_format == "verbose_json":
        return JSONResponse(content={
            "task": "transcribe",
            "language": language or "zh",
            "duration": result["duration"],
            "text": result["text"],
            "segments": [
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "words": seg.get("words", []),
                }
                for i, seg in enumerate(result["segments"])
            ],
        })
    else:
        return JSONResponse(content={"text": result["text"]})


# --- WebSocket: ws://host:port/ws ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streaming WebSocket ASR with dynamic VAD + SPK."""
    from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    vad = DynamicStreamingVAD(_vad_model)
    audio_buffer = np.array([], dtype=np.float32)
    locked_sentences = []
    language = None
    hotwords = None
    use_spk = False
    is_active = False

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                cmd = message["text"].strip()
                if cmd.upper() == "START":
                    vad.reset()
                    audio_buffer = np.array([], dtype=np.float32)
                    locked_sentences = []
                    is_active = True
                    await websocket.send_json({"event": "started"})
                elif cmd.upper().startswith("LANGUAGE:"):
                    language = cmd[9:].strip() or None
                    await websocket.send_json({"event": "language_set", "language": language})
                elif cmd.upper().startswith("HOTWORDS:"):
                    hotwords = [w.strip() for w in cmd[9:].split(",") if w.strip()]
                    await websocket.send_json({"event": "hotwords_set", "hotwords": hotwords})
                elif cmd.upper().startswith("SPK:"):
                    use_spk = cmd[4:].strip().lower() in ("true", "1", "on", "yes")
                    await websocket.send_json({"event": "spk_set", "spk": use_spk})
                elif cmd.upper() == "STOP":
                    if is_active and len(audio_buffer) > 0:
                        # Final: process remaining audio
                        final_segs = vad.finalize()
                        for seg in final_segs:
                            seg_audio = audio_buffer[int(seg[0]*16):int(seg[1]*16)]
                            if len(seg_audio) > 8000:
                                gen_kw = {"max_new_tokens": 500}
                                if language: gen_kw["language"] = language
                                if hotwords: gen_kw["hotwords"] = hotwords
                                res = _engine.generate(inputs=[seg_audio], **gen_kw)
                                if res[0]["text"].strip():
                                    locked_sentences.append({
                                        "text": res[0]["text"], "start": seg[0], "end": seg[1]
                                    })

                        # Handle ongoing speech
                        if vad.is_speaking:
                            end_ms = int(len(audio_buffer) * 1000 / 16000)
                            start_ms = int(vad.current_speech_start) if hasattr(vad, 'current_speech_start') and vad.current_speech_start else 0
                            seg_audio = audio_buffer[int(start_ms*16):]
                            if len(seg_audio) > 8000:
                                gen_kw = {"max_new_tokens": 500}
                                if language: gen_kw["language"] = language
                                if hotwords: gen_kw["hotwords"] = hotwords
                                res = _engine.generate(inputs=[seg_audio], **gen_kw)
                                if res[0]["text"].strip():
                                    locked_sentences.append({
                                        "text": res[0]["text"], "start": start_ms, "end": end_ms
                                    })

                        # SPK: run full clustering on all sentences (only if enabled)
                        if use_spk and locked_sentences and _spk_model is not None:
                            try:
                                from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
                                from funasr.models.campplus.cluster_backend import ClusterBackend
                                vad_segs = [[s["start"]/1000, s["end"]/1000, 
                                            audio_buffer[int(s["start"]*16):int(s["end"]*16)]]
                                           for s in locked_sentences]
                                chunks = sv_chunk(vad_segs)
                                if chunks:
                                    speech_list = [ch[2] for ch in chunks]
                                    spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
                                    import torch as _torch
                                    embs = _torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
                                    cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
                                    labels = cluster(embs.cpu(), oracle_num=None)
                                    if not isinstance(labels, np.ndarray):
                                        labels = np.array(labels)
                                    all_sorted = sorted(chunks, key=lambda x: x[0])
                                    sv_output = postprocess(all_sorted, None, labels, embs.cpu())
                                    spk_sents = [{"text": s["text"], "start": int(s["start"]), "end": int(s["end"])}
                                                for s in locked_sentences]
                                    distribute_spk(spk_sents, sv_output)
                                    for i, ss in enumerate(spk_sents):
                                        locked_sentences[i]["spk"] = ss.get("spk", 0)
                            except Exception as e:
                                logger.warning(f"SPK failed: {e}")

                        await websocket.send_json({
                            "sentences": locked_sentences,
                            "is_final": True,
                            "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                        })
                        is_active = False
                    await websocket.send_json({"event": "stopped"})

            elif "bytes" in message and is_active:
                pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, pcm])

                # Feed VAD
                new_confirmed = vad.feed(torch.from_numpy(pcm).float())
                for seg in new_confirmed:
                    seg_audio = audio_buffer[int(seg[0]*16):int(seg[1]*16)]
                    if len(seg_audio) > 8000:
                        gen_kw = {"max_new_tokens": 500}
                        if language: gen_kw["language"] = language
                        if hotwords: gen_kw["hotwords"] = hotwords
                        res = _engine.generate(inputs=[seg_audio], **gen_kw)
                        if res[0]["text"].strip():
                            locked_sentences.append({
                                "text": res[0]["text"], "start": seg[0], "end": seg[1]
                            })

                # Send partial update
                await websocket.send_json({
                    "sentences": locked_sentences,
                    "is_final": False,
                    "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--qwen-forced-aligner", type=str, default="", help="Optional Qwen3 forced aligner model for timestamp alignment")
    parser.add_argument("--qwen-gpu-memory-utilization", type=float, default=None, help="Optional GPU memory utilization for the Qwen3-ASR vLLM engine")
    parser.add_argument("--qwen-max-model-len", type=int, default=None, help="Optional max model length for the Qwen3-ASR vLLM engine")
    parser.add_argument("--qwen-max-inference-batch-size", type=int, default=128)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=4096)
    parser.add_argument("--preload-models", type=str, default="", help="Comma-separated model routes to preload before serving, for example qwen3-asr-vllm")
    parser.add_argument("--hub", type=str, default="ms")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum files accepted by /asr/batch")
    parser.add_argument("--vad-model", type=str, default="fsmn-vad", help="VAD model name or local path")
    parser.add_argument("--spk-model", type=str, default="iic/speech_eres2netv2_sv_zh-cn_16k-common", help="Speaker model name or local path (set empty to disable)")
    _args = parser.parse_args()

    load_engine(_args)
    preload_models(_args.preload_models)
    uvicorn.run(app, host=_args.host, port=_args.port)
