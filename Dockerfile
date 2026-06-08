# ==========================================
# OmniVoice RunPod Serverless — STT & TTS
# ==========================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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

# Clone OmniVoice Studio (for the omnivoice Python package + model code)
RUN git clone --depth 1 https://github.com/debpalash/OmniVoice-Studio.git /app/repo

# Pin huggingface_hub BEFORE installing anything that depends on it
# (conda base has an old version without is_offline_mode)
RUN uv pip install --system --no-cache "huggingface_hub<0.26"

# Install the omnivoice package (no frontend/extras)
RUN uv pip install --system --no-cache -e /app/repo

# Install RunPod SDK + WhisperX (pyannote.audio comes as dep of whisperx)
RUN uv pip install --system --no-cache \
    runpod \
    whisperx

# Pre-download all models via script
COPY pre_download.py /app/pre_download.py

# Download models with HF_TOKEN if available
RUN --mount=type=secret,id=hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        HF_TOKEN=$(cat /run/secrets/hf_token) python /app/pre_download.py; \
    else \
        python /app/pre_download.py; \
    fi

# Clean up repo source and pre-download script
RUN rm -rf /app/repo /app/pre_download.py

# Copy the RunPod handler
COPY runpod_handler.py /app/runpod_handler.py

CMD ["python", "/app/runpod_handler.py"]
