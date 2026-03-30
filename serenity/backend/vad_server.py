"""WebSocket voice activity detection service for Serenity."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


DEFAULT_VAD_PORT = int(os.getenv("VAD_PORT", "8004"))
DEFAULT_VAD_HOST = os.getenv("VAD_HOST", "0.0.0.0")
VAD_SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
SILENCE_SECONDS = float(os.getenv("VAD_SILENCE_SECONDS", "1.2"))

app = FastAPI(title="Serenity VAD Server", version="1.0.0")
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return the VAD server readiness state."""

    return {
        "ok": True,
        "sample_rate": VAD_SAMPLE_RATE,
        "frame_ms": VAD_FRAME_MS,
        "silence_seconds": SILENCE_SECONDS,
    }


@app.websocket("/vad")
async def vad_websocket(websocket: WebSocket) -> None:
    """Emit a speech-end event after sustained silence in PCM audio frames."""

    await websocket.accept()
    last_voice_time = time.monotonic()
    heard_speech = False
    frame_size = int(VAD_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0) * 2)

    try:
        while True:
            chunk = await websocket.receive_bytes()
            if len(chunk) < frame_size:
                continue

            speech_detected = False
            for index in range(0, len(chunk) - frame_size + 1, frame_size):
                frame = chunk[index : index + frame_size]
                if vad.is_speech(frame, VAD_SAMPLE_RATE):
                    speech_detected = True
                    heard_speech = True
                    last_voice_time = time.monotonic()

            if heard_speech and not speech_detected and (time.monotonic() - last_voice_time) >= SILENCE_SECONDS:
                await websocket.send_text(json.dumps({"event": "speech_end"}))
                heard_speech = False
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.vad_server:app", host=DEFAULT_VAD_HOST, port=DEFAULT_VAD_PORT, reload=False)
