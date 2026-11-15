# syntax=docker/dockerfile:1
#
# OneArt GPU runtime image (docs/milestones/M11-gpu-pool.md T11.15). Build with `npm run gpu:image` (scripts/gpu-image.sh),
# context = gpu/ (so gpu/.dockerignore applies):
#   docker build -f gpu/image/Dockerfile --target runtime --build-arg IMAGE_VERSION=<git sha> -t oneart-gpu:<git sha> gpu
#
# Single source of truth for versions: gpu/remote/bootstrap.py (ComfyUI and node-pack commits, wav2vec patch),
# gpu/remote/lora.py (ai-toolkit commit, torch 2.13 venv). Every install step below calls those files, so the image can
# never drift from what a bare server installs. The only version that is not pinned there is SageAttention (see ARG).
#
# Secrets: none. No token, key or credential is read, copied or baked in (the model weights we need are ungated).
# Targets: `runtime` (default: video tools only, no weights, no AI-Toolkit), `runtime-lora` (adds AI-Toolkit for LoRA
# training) and `runtime-h3` (adds the H3 and klein weights; build it only if exp-025 says it pays off).

# Pinned base: the same image a bare server boots today (docs/07 §ایمیج).
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime AS image-base

# Empty = latest SageAttention at build time; the resolved version is recorded in /opt/oneart/IMAGE_MANIFEST.json.
# Set it (--build-arg SAGEATTENTION_VERSION=x.y.z) to freeze a release once exp confirms one.
ARG SAGEATTENTION_VERSION=""
ARG IMAGE_VERSION="dev"
# Written to /opt/oneart/AGENT_VERSION for the health check; scripts/gpu-image.sh passes the constant of gpu/remote/agent.py.
ARG AGENT_VERSION="dev"

# The base is Ubuntu 24.04 with the system Python (PEP 668, EXTERNALLY-MANAGED); the container is ours alone, so pip may
# install system-wide, the same as the bare-server path (orchestrator.service.ts).
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    COMFY_DIR=/opt/oneart/ComfyUI \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends git ffmpeg rsync curl aria2 libgl1 libglib2.0-0 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Our scripts first (they drive every install step). bootstrap.py runs at build time, so the scripts are copied before it.
COPY remote/ /opt/oneart/bin/
COPY tools/validate_local.py tools/dry_run.py tools/build_local.py tools/smoke_plan.py /opt/oneart/bin/
COPY image/smoke.sh /opt/oneart/bin/smoke.sh

# Build-time scratch home: ai-toolkit must live at a fixed path (a venv is not relocatable), so its checkout is built
# under /opt/oneart/home/repos and linked into the runtime OA_HOME by bootstrap.py (image_link_repos).
WORKDIR /opt/oneart/bin
RUN pip install --no-cache-dir -q -U 'huggingface_hub[hf_transfer]' hf_transfer

RUN OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models python3 bootstrap.py --image-build comfy
RUN OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models python3 bootstrap.py --image-build nodes
RUN OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models SAGEATTENTION_VERSION="${SAGEATTENTION_VERSION}" python3 bootstrap.py --image-build sage
# AI-Toolkit (LoRA training, its own venv with a second torch, ~4.9 GB) is NOT in the default image (owner, 15 Nov):
# a video build never uses it. A LoRA job installs it on first use (lora.setup()), or build the runtime-lora target.

# Records the real commits and versions, then marks the image ready. Build logs and scratch dirs are removed.
RUN OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models IMAGE_VERSION="${IMAGE_VERSION}" python3 bootstrap.py --image-build finish \
 && printf '%s\n' "${AGENT_VERSION}" > /opt/oneart/AGENT_VERSION \
 && rm -rf /opt/oneart/home/logs /opt/oneart/home/models /opt/oneart/home/inputs /opt/oneart/home/out /opt/oneart/home/tmp \
           /root/.cache /tmp/* \
 && find /opt/oneart -name __pycache__ -type d -prune -exec rm -rf {} +

LABEL org.opencontainers.image.title="oneart-gpu" org.opencontainers.image.version="${IMAGE_VERSION}"
ENV PATH=/opt/oneart/bin:${PATH}
WORKDIR /workspace

# ---------------------------------------------------------------- runtime-h3: ONLY if exp-025 says so
# Adds the weights of the H3 pipeline (MiniMax H3 + FLUX.2 klein 4B), ungated on Hugging Face, so no token is used.
# Large image (tens of GB): the rented server then skips the download. Never the default target.
FROM image-base AS runtime-h3
RUN OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models python3 /opt/oneart/bin/download.py minimax-h3 flux2-klein-4b \
 && rm -rf /opt/oneart/home/logs /opt/oneart/home/models/_hf /root/.cache

# ---------------------------------------------------------------- runtime-lora: video tools + AI-Toolkit
# For servers that train LoRAs: adds AI-Toolkit (pinned in gpu/remote/lora.py) and rewrites the manifest so selfcheck
# verifies it too. Not the default; build it with `npm run gpu:image -- runtime-lora`.
FROM image-base AS runtime-lora
ARG IMAGE_VERSION="dev"
WORKDIR /opt/oneart/bin
RUN OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models python3 bootstrap.py --image-build aitoolkit \
 && OA_HOME=/opt/oneart/home OA_MODELS=/opt/oneart/home/models IMAGE_VERSION="${IMAGE_VERSION}" python3 bootstrap.py --image-build finish \
 && rm -rf /opt/oneart/home/logs /opt/oneart/home/tmp /root/.cache /tmp/* \
 && find /opt/oneart -name __pycache__ -type d -prune -exec rm -rf {} +
WORKDIR /workspace

# ---------------------------------------------------------------- runtime: default, no weights (OQ-77)
# Last stage on purpose: a plain `docker build` without --target builds this one.
FROM image-base AS runtime
