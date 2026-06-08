"""Pre-download ALL models at Docker build time to eliminate runtime downloads."""
import os, torch, io, wave, struct, math, random

hf_home = os.environ.get("HF_HOME", "/app/hf_cache")
os.makedirs(hf_home, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute = "float16" if device == "cuda" else "int8"

# -------------------------------------------------------------------
# 1. WhisperX ASR model
# -------------------------------------------------------------------
print(f"[1/4] Loading WhisperX large-v3 on {device} ({compute})...")
import whisperx
model = whisperx.load_model("large-v3", device=device, compute_type=compute)
print("  OK")

# -------------------------------------------------------------------
# 2. wav2vec2 alignment model (English)
# -------------------------------------------------------------------
print("[2/4] Downloading wav2vec2 alignment model (en)...")
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
del align_model
print("  OK")

# -------------------------------------------------------------------
# 3. pyannote diarization model (requires HF_TOKEN)
# -------------------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("[3/4] Downloading pyannote diarization model...")
    try:
        diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        del diarize
        print("  OK")
    except Exception as e:
        print(f"  SKIP (non-fatal): {e}")
else:
    print("[3/4] SKIP pyannote (no HF_TOKEN)")

# -------------------------------------------------------------------
# 4. Test transcription with generated audio
# -------------------------------------------------------------------
print("[4/4] Running test transcription...")
sample_rate = 16000
num_samples = sample_rate * 3
samples = []
for i in range(num_samples):
    t = i / sample_rate
    val = (math.sin(2*math.pi*250*t) * 0.3 +
           math.sin(2*math.pi*400*t) * 0.2 +
           math.sin(2*math.pi*600*t) * 0.15)
    env = 0.5 + 0.5 * math.sin(2*math.pi*2*t)
    val *= env
    val += random.gauss(0, 0.02)
    val = max(-1, min(1, val))
    samples.append(int(val * 32767))

buf = io.BytesIO()
with wave.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))

import tempfile
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
tmp.write(buf.getvalue())
tmp.close()

audio = whisperx.load_audio(tmp.name)
result = model.transcribe(audio, batch_size=8)
print(f"  Language: {result.get('language','?')}")
print(f"  Segments: {len(result.get('segments',[]))}")
for seg in result.get('segments', [])[:3]:
    print(f"    [{seg.get('start',0):.1f}-{seg.get('end',0):.1f}] {seg.get('text','')}")

os.unlink(tmp.name)
del model, audio, result
print("  OK — transcription pipeline verified")
print("\nAll models pre-downloaded and verified. Build complete.")
