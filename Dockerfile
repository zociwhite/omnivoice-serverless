# ==========================================
# OmniVoice RunPod Serverless — STT & TTS
# ==========================================
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

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

# Clone OmniVoice Studio
RUN git clone --depth 1 https://github.com/debpalash/OmniVoice-Studio.git /app/repo

# Install the omnivoice package (regular install, not editable)
RUN uv pip install --system --no-cache /app/repo

# Install RunPod SDK + WhisperX (pyannote.audio comes as dep of whisperx)
RUN uv pip install --system --no-cache \
    runpod \
    whisperx


# Clean up repo source (models download at runtime for reliable builds)
RUN rm -rf /app/repo

# Copy the RunPod handler
COPY runpod_handler.py /app/runpod_handler.py

CMD ["python", "/app/runpod_handler.py"]
