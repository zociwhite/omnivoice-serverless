# ==========================================
# OmniVoice RunPod Serverless — STT & TTS
# ==========================================
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast pip resolution
RUN pip install --no-cache-dir uv

# Clone OmniVoice Studio (for the omnivoice Python package + model code)
RUN git clone --depth 1 https://github.com/debpalash/OmniVoice-Studio.git /app/repo

# Install the omnivoice package (no frontend/extras)
RUN uv pip install --system --no-cache -e /app/repo

# Install RunPod SDK + WhisperX (pyannote.audio comes as dep of whisperx)
RUN uv pip install --system --no-cache \
    runpod \
    whisperx

# Pre-download OmniVoice model weights (TTS)
RUN python -c "
from omnivoice import OmniVoice
import torch
model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', load_asr=False)
model.eval()
print('OmniVoice model downloaded and cached.')
"

# Pre-download WhisperX large-v3 (STT)
RUN python -c "
import whisperx
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisperx.load_model('large-v3', device=device, compute_type='float16' if device == 'cuda' else 'int8')
print('WhisperX model downloaded and cached.')
del model
"

# Pre-download pyannote diarization model
# Requires HF_TOKEN at build time — use BuildKit --secret
RUN --mount=type=secret,id=hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
        python -c "
import torch
from pyannote.audio import Pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.environ['HF_TOKEN'])
pipeline.to(torch.device(device))
print('pyannote model downloaded and cached.')
del pipeline
"; \
    else \
        echo 'No HF_TOKEN secret provided — skipping pyannote pre-download (will download at runtime if HF_TOKEN is set)'; \
    fi

# Clean up repo source (keep only installed package)
RUN rm -rf /app/repo

# Copy the RunPod handler
COPY runpod_handler.py /app/runpod_handler.py

# RunPod serverless — the handler calls runpod.serverless.start() internally
CMD ["python", "/app/runpod_handler.py"]
