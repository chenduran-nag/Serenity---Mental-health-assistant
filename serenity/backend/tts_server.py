"""Local Coqui TTS service for Serenity voice replies."""

from __future__ import annotations

import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from TTS.api import TTS


DEFAULT_TTS_PORT = int(os.getenv("TTS_PORT", "8003"))
DEFAULT_TTS_HOST = os.getenv("TTS_HOST", "0.0.0.0")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/glow-tts")

app = FastAPI(title="Serenity TTS Server", version="1.0.0")
_tts: Any | None = None


class SpeechRequest(BaseModel):
    """Request payload for text-to-speech synthesis."""

    text: str = Field(..., min_length=1, max_length=2000)


def _enhance_prosody(text: str) -> str:
    """Insert lightweight pause hints for more natural phrasing."""

    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r",\s*", ", ... ", normalized)
    normalized = re.sub(r"\.\s*", ". ... ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@app.on_event("startup")
async def startup_event() -> None:
    """Load the TTS model during startup."""

    global _tts
    _tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=False)


@app.post("/speak")
async def speak(payload: SpeechRequest) -> StreamingResponse:
    """Convert text into spoken audio and stream a WAV response."""

    if _tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not loaded.")

    enhanced_text = _enhance_prosody(payload.text)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            temp_path = Path(handle.name)
        _tts.tts_to_file(text=enhanced_text, file_path=str(temp_path))
        if not temp_path.exists():
            raise HTTPException(status_code=500, detail="TTS synthesis did not create an audio file.")
        audio_bytes = temp_path.read_bytes()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {type(exc).__name__}.") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return StreamingResponse(iter([audio_bytes]), media_type="audio/wav")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return the TTS server readiness state."""

    return {
        "ok": _tts is not None,
        "model_name": TTS_MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.tts_server:app", host=DEFAULT_TTS_HOST, port=DEFAULT_TTS_PORT, reload=False)
