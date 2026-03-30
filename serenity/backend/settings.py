"""Configuration helpers for Serenity backend services."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class ServiceSettings:
    """Centralized runtime configuration for backend services."""

    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = _get_env_int("BACKEND_PORT", 8000)
    model_server_url: str = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8001")
    stt_server_url: str = os.getenv("STT_SERVER_URL", "http://127.0.0.1:8002")
    tts_server_url: str = os.getenv("TTS_SERVER_URL", "http://127.0.0.1:8003")
    vad_server_url: str = os.getenv("VAD_SERVER_URL", "ws://127.0.0.1:8004/vad")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./serenity.db")
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "*")


SETTINGS = ServiceSettings()
