"""Local Whisper-based speech-to-text service for Serenity."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


DEFAULT_STT_PORT = int(os.getenv("STT_PORT", "8002"))
DEFAULT_STT_HOST = os.getenv("STT_HOST", "0.0.0.0")
EXPLICIT_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME")

app = FastAPI(title="Serenity STT Server", version="1.0.0")
_model: Any | None = None
_model_name: str | None = None


class TranscriptionResponse(BaseModel):
    """Response payload for an audio transcription."""

    transcript: str


def _detect_total_ram_gb() -> float:
    """Return total system RAM in gigabytes using platform-safe fallbacks."""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            """Windows memory status struct."""

            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MemoryStatus()
        memory_status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        return float(memory_status.ullTotalPhys) / (1024**3)

    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return float(pages * page_size) / (1024**3)


def _select_model_name() -> str:
    """Choose the Whisper model size based on RAM unless explicitly overridden."""

    if EXPLICIT_MODEL_NAME:
        return EXPLICIT_MODEL_NAME
    return "small" if _detect_total_ram_gb() > 8 else "base"


def _load_model() -> Any:
    """Load the Whisper model once for process reuse."""

    global _model_name
    _model_name = _select_model_name()
    return whisper.load_model(_model_name)


@app.on_event("startup")
async def startup_event() -> None:
    """Load the speech model during application startup."""

    global _model
    _model = _load_model()


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)) -> TranscriptionResponse:
    """Transcribe an uploaded WAV or WebM file into text."""

    if _model is None:
        raise HTTPException(status_code=503, detail="Whisper model is not loaded.")

    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to read uploaded audio.") from exc

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
            temp_path = Path(handle.name)

        result = _model.transcribe(str(temp_path), fp16=False)
        transcript = str(result.get("text", "")).strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {type(exc).__name__}.") from exc
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech could be transcribed from the audio.")

    return TranscriptionResponse(transcript=transcript)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return the STT server readiness state."""

    return {
        "ok": _model is not None,
        "model_name": _model_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.stt_server:app", host=DEFAULT_STT_HOST, port=DEFAULT_STT_PORT, reload=False)
