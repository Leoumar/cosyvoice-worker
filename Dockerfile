FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System packages (sox + ffmpeg are needed for audio handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential ffmpeg sox libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (conda-forge by default) - CosyVoice needs conda to install pynini
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

RUN /opt/conda/bin/conda create -y -n cosyvoice python=3.10 \
    && /opt/conda/bin/conda install -y -n cosyvoice -c conda-forge pynini==2.1.5 \
    && /opt/conda/bin/conda clean -afy

# Use the cosyvoice environment for everything below
ENV PATH=/opt/conda/envs/cosyvoice/bin:$PATH

WORKDIR /app
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

WORKDIR /app/CosyVoice
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir runpod huggingface_hub

# Bake the model weights into the image so workers start fast (no download at boot)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

WORKDIR /app
# Build context is the repo root, so the handler lives in the cosyvoice-worker/ subfolder
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]