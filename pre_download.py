"""Pre-download WhisperX large-v3 at Docker build time.
OmniVoice and pyannote are downloaded at runtime (acceptable cold start)."""
import os, torch

hf_home = os.environ.get("HF_HOME", "/app/hf_cache")
os.makedirs(hf_home, exist_ok=True)

print("Downloading WhisperX large-v3 model...")
import whisperx
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device=device,
                            compute_type="float16" if device == "cuda" else "int8")
del model
print("WhisperX model cached.")
print("Pre-download complete.")
