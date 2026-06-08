# ==========================================
# OmniVoice RunPod Serverless — STT & TTS
# ==========================================
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache
# PyTorch 2.6+ defaults weights_only=True; our models use pickle
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

# Pin huggingface_hub BEFORE transformers (avoid use_auth_token removal)
RUN uv pip install --system --no-cache "huggingface_hub<0.26"

# Install omnivoice
RUN uv pip install --system --no-cache -e /app/repo

# Re-pin huggingface_hub (transformers may upgrade it)
RUN uv pip install --system --no-cache "huggingface_hub<0.26"

# Install RunPod SDK + WhisperX
RUN uv pip install --system --no-cache \
    runpod \
    whisperx

# Pre-download all models
COPY pre_download.py /app/pre_download.py

RUN --mount=type=secret,id=hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        HF_TOKEN=$(cat /run/secrets/hf_token) python /app/pre_download.py; \
    else \
        python /app/pre_download.py; \
    fi

# Clean up
RUN rm -rf /app/repo /app/pre_download.py

COPY runpod_handler.py /app/runpod_handler.py

CMD ["python", "/app/runpod_handler.py"]
