"""Primary FastAPI backend for the Serenity mental health assistant."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from backend.crisis_detector import CRISIS_RESOURCE_MESSAGE, detect_crisis
from backend.database import delete_history, fetch_history, init_db, save_messages


def _service_url(env_name: str, default: str) -> str:
    """Return a service base URL from environment or fallback."""

    return os.getenv(env_name, default).rstrip("/")


def _session_rate_key(request: Request) -> str:
    """Generate a session-aware rate limiting key."""

    session_id = getattr(request.state, "session_id", None)
    return str(session_id or get_remote_address(request))


API_PORT = int(os.getenv("SERENITY_BACKEND_PORT", "8000"))
MODEL_SERVER_URL = _service_url("SERENITY_MODEL_SERVER_URL", "http://127.0.0.1:8001")
STT_SERVER_URL = _service_url("SERENITY_STT_SERVER_URL", "http://127.0.0.1:8002")
TTS_SERVER_URL = _service_url("SERENITY_TTS_SERVER_URL", "http://127.0.0.1:8003")
SERVICE_TIMEOUT_SECONDS = float(os.getenv("SERENITY_HTTP_TIMEOUT", "120"))

limiter = Limiter(key_func=_session_rate_key, default_limits=[])
app = FastAPI(title="Serenity Backend", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatMessage(BaseModel):
    """Single role/content chat message."""

    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatTextRequest(BaseModel):
    """Payload for text chat conversations."""

    session_id: str | None = Field(default=None)
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatTextResponse(BaseModel):
    """Response payload for text chat conversations."""

    reply: str
    session_id: str
    timestamp: str
    crisis_detected: bool = False
    crisis_message: str | None = None


def _ensure_session_id(session_id: str | None) -> str:
    """Create a UUID-based session ID when one is not provided."""

    return session_id or str(uuid.uuid4())


def _format_history(messages: list[ChatMessage], latest_message: str) -> str:
    """Format a simple chat transcript for the model server."""

    lines: list[str] = []
    for item in messages:
        speaker = "User" if item.role.lower() == "user" else "Assistant"
        lines.append(f"{speaker}: {item.content.strip()}")
    lines.append(f"User: {latest_message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _apply_rate_limit(request: Request, session_id: str) -> None:
    """Enforce the configured rate limit for a session."""

    request.state.session_id = session_id
    limit_item = limiter._limiter.parse("60/minute")
    if not limiter.limiter.hit(limit_item, session_id):
        raise RateLimitExceeded(detail="Rate limit exceeded for this session.")


async def _generate_reply(prompt: str) -> str:
    """Call the model server to generate a response."""

    payload: dict[str, Any] = {"prompt": prompt, "max_tokens": 256, "temperature": 0.7}
    try:
        async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{MODEL_SERVER_URL}/generate", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Model server returned {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Model server is unavailable.") from exc

    reply = str(data.get("response", "")).strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Model server returned an empty response.")
    return reply


async def _transcribe_audio(upload: UploadFile) -> str:
    """Send audio to the STT service and return the transcript."""

    file_bytes = await upload.read()
    files = {"audio_file": (upload.filename or "audio.wav", file_bytes, upload.content_type or "audio/wav")}
    try:
        async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{STT_SERVER_URL}/transcribe", files=files)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"STT server returned {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="STT server is unavailable.") from exc

    transcript = str(data.get("transcript", "")).strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech transcript could be produced.")
    return transcript


async def _speak_text(text: str) -> tuple[bytes, str]:
    """Send text to the TTS service and return synthesized audio bytes."""

    try:
        async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{TTS_SERVER_URL}/speak", json={"text": text})
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"TTS server returned {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="TTS server is unavailable.") from exc

    return response.content, response.headers.get("content-type", "audio/wav")


async def _service_health(name: str, base_url: str) -> dict[str, Any]:
    """Probe an upstream service health endpoint."""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            return {"service": name, "ok": True, "details": response.json()}
    except Exception as exc:
        return {"service": name, "ok": False, "details": str(exc)}


async def _chat_text_impl(request: Request, payload: ChatTextRequest) -> ChatTextResponse:
    """Shared implementation for text chat logic."""

    session_id = _ensure_session_id(payload.session_id)
    _apply_rate_limit(request, session_id)

    crisis = detect_crisis(payload.message)
    prompt = _format_history(payload.history, payload.message)
    reply = await _generate_reply(prompt)

    if crisis.crisis_detected:
        reply = f"{reply}\n\n{CRISIS_RESOURCE_MESSAGE}"

    save_messages(session_id, [("user", payload.message), ("assistant", reply)])
    timestamp = datetime.now(timezone.utc).isoformat()
    return ChatTextResponse(
        reply=reply,
        session_id=session_id,
        timestamp=timestamp,
        crisis_detected=crisis.crisis_detected,
        crisis_message=crisis.resource_message,
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize persistent services at app startup."""

    init_db()


@app.post("/chat/text", response_model=ChatTextResponse)
async def chat_text(request: Request, payload: ChatTextRequest) -> ChatTextResponse:
    """Handle a text chat request."""

    return await _chat_text_impl(request, payload)


@app.post("/chat/voice")
async def chat_voice(
    request: Request,
    audio_file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
) -> Response:
    """Handle a voice chat round trip from transcription through synthesis."""

    transcript = await _transcribe_audio(audio_file)
    text_response = await _chat_text_impl(
        request,
        ChatTextRequest(session_id=session_id, message=transcript, history=[]),
    )
    audio_bytes, media_type = await _speak_text(text_response.reply)
    headers = {
        "X-Transcript": transcript,
        "X-Session-Id": text_response.session_id,
        "X-Reply-Text": text_response.reply,
        "X-Crisis-Detected": str(text_response.crisis_detected).lower(),
    }
    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str) -> JSONResponse:
    """Return persisted chat history for a session."""

    history = fetch_history(session_id)
    return JSONResponse(
        content={
            "session_id": session_id,
            "messages": [
                {
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                }
                for row in history
            ],
        }
    )


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str) -> JSONResponse:
    """Delete persisted chat history for a session."""

    deleted = delete_history(session_id)
    return JSONResponse(content={"session_id": session_id, "deleted": deleted})


@app.get("/health")
async def health() -> JSONResponse:
    """Return backend and upstream service health."""

    services = await _gather_service_health()
    overall_ok = all(item["ok"] for item in services.values())
    status_code = 200 if overall_ok else 503
    return JSONResponse(content={"ok": overall_ok, "services": services}, status_code=status_code)


async def _gather_service_health() -> dict[str, dict[str, Any]]:
    """Collect health information for all dependent local services."""

    probes = {
        "model": MODEL_SERVER_URL,
        "stt": STT_SERVER_URL,
        "tts": TTS_SERVER_URL,
    }
    services: dict[str, dict[str, Any]] = {}
    for name, base_url in probes.items():
        services[name] = await _service_health(name, base_url)
    return services


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=API_PORT, reload=False)
