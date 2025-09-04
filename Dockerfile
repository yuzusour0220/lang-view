FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for Python, OpenCV, and video I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git ffmpeg libgl1 libglib2.0-0 \
    build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Use repo requirements (includes PyTorch/cu121 via extra-index-url)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && pip install scipy==1.11.4

# Keep repo on PYTHONPATH when bind-mounting the project
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default entrypoint runs python; pass your script/args after the image name
ENTRYPOINT ["python3"]

