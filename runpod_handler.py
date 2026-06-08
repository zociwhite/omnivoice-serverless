"""
OmniVoice RunPod Serverless Handler
=====================================
Single handler with TASK_TYPE branching: STT (WhisperX) or TTS (OmniVoice).
All models pre-downloaded at Docker build time, lazy-loaded on first request.

Environment:
    TASK_TYPE       "stt" (default) or "tts"
    HF_TOKEN        HuggingFace token (for pyannote diarization)
"""
import base64
import io
import logging
import os
import tempfile
import time

import runpod
import torch
import torchaudio
import whisperx

logger = logging.getLogger(__name__)

TASK_TYPE = os.environ.get("TASK_TYPE", "stt").lower()


# ---------------------------------------------------------------------------
# Lazy model singletons (loaded once, kept in VRAM across requests)
# ---------------------------------------------------------------------------
_whisperx_pipeline = {"model": None, "diarize": None}
_omnivoice_model = None


def _load_whisperx():
    """Lazy-load WhisperX ASR + wav2vec2 aligner + pyannote diarization."""
    if _whisperx_pipeline["model"] is not None:
        return _whisperx_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"

    logger.info("Loading WhisperX large-v3 on %s (%s)", device, compute)
    model = whisperx.load_model("large-v3-turbo", device=device, compute_type=compute)

    # Lazy-load diarization model (pyannote)
    diarize = None
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            logger.info("Loading pyannote speaker diarization...")
            diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        except Exception as e:
            logger.warning("Diarization load failed (proceeding without): %s", e)

    _whisperx_pipeline["model"] = model
    _whisperx_pipeline["diarize"] = diarize
    return _whisperx_pipeline


def _load_omnivoice():
    """Lazy-load OmniVoice TTS model."""
    global _omnivoice_model
    if _omnivoice_model is not None:
        return _omnivoice_model

    from omnivoice import OmniVoice

    logger.info("Loading OmniVoice model...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        load_asr=False,
        torch_dtype=dtype,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    _omnivoice_model = model
    return model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode_audio(job_input: dict) -> str:
    """Extract audio from job input (base64 or URL) and save to temp file.

    Returns path to the saved audio file.
    """
    audio_data = job_input.get("audio") or job_input.get("audio_base64")
    audio_url = job_input.get("audio_url")

    if audio_data:
        raw = base64.b64decode(audio_data)
        suffix = job_input.get("audio_format", ".wav")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(raw)
        tmp.close()
        return tmp.name

    if audio_url:
        import urllib.request
        suffix = os.path.splitext(audio_url.split("?")[0])[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        urllib.request.urlretrieve(audio_url, tmp.name)
        return tmp.name

    raise ValueError("Missing audio input: provide 'audio' (base64) or 'audio_url'")


def audio_to_base64(wav_tensor, sample_rate: int = 24000) -> str:
    """Convert a PyTorch audio tensor to base64-encoded WAV."""
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, wav_tensor.cpu(), sample_rate, format="wav")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# STT handler
# ---------------------------------------------------------------------------

def _handle_stt(job_input: dict) -> dict:
    """Speech-to-Text: transcribe → align → diarize → assign speakers."""
    pipe = _load_whisperx()
    model = pipe["model"]
    diarize = pipe["diarize"]

    audio_path = _decode_audio(job_input)

    # 1. Transcribe
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=job_input.get("batch_size", 8))
    lang = result.get("language", "en")

    # 2. Forced alignment (wav2vec2)
    try:
        align_model, align_metadata = whisperx.load_align_model(language_code=lang, device=model.device)
        result = whisperx.align(result["segments"], align_model, align_metadata, audio, model.device)
    except Exception as e:
        logger.warning("Alignment skipped: %s", e)

    # 3. Diarization (if available)
    if diarize is not None:
        try:
            diarize_segments = diarize(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            logger.warning("Diarization skipped: %s", e)

    # 4. Build output
    segments_out = []
    for seg in result.get("segments", []):
        segments_out.append({
            "text": seg.get("text", ""),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "speaker": seg.get("speaker"),
            "words": [
                {"word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
                for w in (seg.get("words") or [])
            ],
        })

    try:
        os.unlink(audio_path)
    except Exception:
        pass

    return {"segments": segments_out, "language": lang}


# ---------------------------------------------------------------------------
# TTS handler
# ---------------------------------------------------------------------------

def _handle_tts(job_input: dict) -> dict:
    """Text-to-Speech: voice clone, voice design, or auto voice."""
    model = _load_omnivoice()

    text = job_input.get("text", "")
    if not text:
        raise ValueError("Missing 'text' field for TTS")

    language = job_input.get("language")
    instruct = job_input.get("instruct")
    ref_audio_b64 = job_input.get("ref_audio") or job_input.get("audio_base64")
    ref_text = job_input.get("ref_text")
    speed = job_input.get("speed", 1.0)
    num_step = job_input.get("num_step", 16)
    guidance_scale = job_input.get("guidance_scale", 2.0)

    ref_audio_path = None
    if ref_audio_b64:
        raw = base64.b64decode(ref_audio_b64)
        ref_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(ref_audio_path, "wb") as f:
            f.write(raw)

    try:
        audios = model.generate(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            instruct=instruct,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
        wav = audios[0]  # (1, T)
        audio_b64 = audio_to_base64(wav, sample_rate=model.sampling_rate)
    finally:
        if ref_audio_path:
            try:
                os.unlink(ref_audio_path)
            except Exception:
                pass

    return {
        "audio": audio_b64,
        "sample_rate": model.sampling_rate,
        "duration": wav.shape[-1] / model.sampling_rate,
    }


# ---------------------------------------------------------------------------
# RunPod entry point
# ---------------------------------------------------------------------------

def handler(job):
    """Main RunPod handler — dispatches by TASK_TYPE."""
    job_input = job["input"]
    start = time.time()

    try:
        if TASK_TYPE == "stt":
            result = _handle_stt(job_input)
        elif TASK_TYPE == "tts":
            result = _handle_tts(job_input)
        else:
            raise ValueError(f"Unknown TASK_TYPE: {TASK_TYPE!r} (use 'stt' or 'tts')")

        elapsed = time.time() - start
        result["processing_time_s"] = round(elapsed, 2)
        return result

    except Exception as e:
        logger.exception("Handler failed")
        return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runpod.serverless.start({"handler": handler})
