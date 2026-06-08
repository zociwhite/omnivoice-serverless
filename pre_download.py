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

# 3. pyannote diarization is skipped at build time because it uses
# use_auth_token (deprecated in huggingface_hub>=0.24). It will be
# downloaded at runtime via HF_TOKEN env var when diarization is requested.
print("pyannote diarization will be downloaded at runtime (HF_TOKEN required).")
