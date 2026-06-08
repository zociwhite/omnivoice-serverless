"""Pre-download all models at Docker build time."""
import os, torch

# 1. OmniVoice TTS model
print("Downloading OmniVoice model...")
from omnivoice import OmniVoice
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", load_asr=False)
model.eval()
del model
print("OmniVoice model cached.")

# 2. WhisperX large-v3
print("Downloading WhisperX large-v3...")
import whisperx
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device=device,
                            compute_type="float16" if device == "cuda" else "int8")
del model
print("WhisperX model cached.")

# 3. pyannote diarization (requires HF_TOKEN)
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    print("Downloading pyannote diarization...")
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device(device))
    del pipeline
    print("pyannote model cached.")
else:
    print("No HF_TOKEN — skipping pyannote pre-download.")
