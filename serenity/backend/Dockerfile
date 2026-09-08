FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch lands in its own layer, installed by a command identical for every
# service, so the image store keeps one copy rather than one per service.
# CPU wheels are the default: docker-compose.yml defaults NVIDIA_VISIBLE_DEVICES
# to none, so the CUDA runtime bundled into the PyPI wheels is dead weight worth
# roughly 3 GB per image. Override TORCH_INDEX_URL to build GPU images.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip && pip install --index-url ${TORCH_INDEX_URL} "torch>=2.1,<3" torchaudio

# Each service then installs only its own stack, so the Whisper, Coqui, and
# training pins never resolve against one another in a single environment.
ARG REQUIREMENTS=requirements.txt

COPY requirements*.txt /tmp/deps/
RUN pip install -r /tmp/deps/${REQUIREMENTS}

COPY . /app/serenity
WORKDIR /app/serenity

ENV PYTHONPATH=/app/serenity
