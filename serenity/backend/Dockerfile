FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Each service installs only its own stack, so the Whisper, Coqui, and training
# pins never have to resolve against one another in a single environment.
ARG REQUIREMENTS=requirements.txt

COPY requirements*.txt /tmp/deps/
RUN pip install --upgrade pip && pip install -r /tmp/deps/${REQUIREMENTS}

COPY . /app/serenity
WORKDIR /app/serenity

ENV PYTHONPATH=/app/serenity
