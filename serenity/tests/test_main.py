"""API-level tests for the Serenity backend orchestration layer."""

from __future__ import annotations

import base64
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from limits import RateLimitItemPerMinute

from backend import main as main_module
from backend.crisis_detector import CRISIS_RESOURCE_MESSAGE


# A reply shaped like real model output: multiple lines, and non-latin-1
# characters. Both are unrepresentable in an HTTP header value.
MULTILINE_REPLY = """That sounds genuinely heavy, and I am glad you said it out loud.

Would it help to talk through what today looked like?"""

UNICODE_TRANSCRIPT = "Estoy muy triste — ¿puedes ayudarme?"

BINARY_AUDIO = bytes(range(256)) * 4


@pytest.fixture()
def client():
    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture()
def session_id():
    return f"test-{uuid.uuid4().hex}"


def _stub_generate(monkeypatch, reply=MULTILINE_REPLY):
    async def _fake(prompt):
        return reply

    monkeypatch.setattr(main_module, "_generate_reply", _fake)


def _stub_generate_failure(monkeypatch, status_code=503):
    async def _fake(prompt):
        raise HTTPException(status_code=status_code, detail="Model server is unavailable.")

    monkeypatch.setattr(main_module, "_generate_reply", _fake)


def _stub_voice(monkeypatch, transcript=UNICODE_TRANSCRIPT):
    async def _fake_transcribe(upload):
        return transcript

    async def _fake_speak(text):
        return BINARY_AUDIO, "audio/wav"

    monkeypatch.setattr(main_module, "_transcribe_audio", _fake_transcribe)
    monkeypatch.setattr(main_module, "_speak_text", _fake_speak)


# --- text chat -----------------------------------------------------------


def test_chat_text_returns_reply(client, monkeypatch, session_id):
    """Regression: this path used to raise AttributeError inside the rate limiter."""

    _stub_generate(monkeypatch, "I hear you.")
    response = client.post(
        "/chat/text",
        json={"session_id": session_id, "message": "I feel anxious today.", "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "I hear you."
    assert body["session_id"] == session_id
    assert body["crisis_detected"] is False
    assert body["crisis_message"] is None


def test_chat_text_generates_session_id_when_absent(client, monkeypatch):
    _stub_generate(monkeypatch, "I hear you.")
    response = client.post("/chat/text", json={"message": "Hello.", "history": []})
    assert response.status_code == 200
    assert uuid.UUID(response.json()["session_id"])


def test_model_failure_propagates_for_non_crisis(client, monkeypatch, session_id):
    _stub_generate_failure(monkeypatch)
    response = client.post(
        "/chat/text",
        json={"session_id": session_id, "message": "Work was rough.", "history": []},
    )
    assert response.status_code == 503


# --- crisis safety -------------------------------------------------------


def test_crisis_message_is_appended(client, monkeypatch, session_id):
    _stub_generate(monkeypatch, "I am really glad you told me.")
    response = client.post(
        "/chat/text",
        json={"session_id": session_id, "message": "I want to die.", "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crisis_detected"] is True
    assert body["crisis_message"] == CRISIS_RESOURCE_MESSAGE
    assert CRISIS_RESOURCE_MESSAGE in body["reply"]


def test_crisis_resources_survive_model_failure(client, monkeypatch, session_id):
    """Safety regression: a dead model server must not swallow the hotline message."""

    _stub_generate_failure(monkeypatch)
    response = client.post(
        "/chat/text",
        json={"session_id": session_id, "message": "I am going to kill myself.", "history": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crisis_detected"] is True
    assert body["reply"] == CRISIS_RESOURCE_MESSAGE
    assert body["crisis_message"] == CRISIS_RESOURCE_MESSAGE


# --- rate limiting -------------------------------------------------------


def test_rate_limit_returns_429_when_exceeded(client, monkeypatch, session_id):
    _stub_generate(monkeypatch, "I hear you.")
    monkeypatch.setattr(main_module, "RATE_LIMIT_ITEM", RateLimitItemPerMinute(2))

    payload = {"session_id": session_id, "message": "Hello.", "history": []}
    assert client.post("/chat/text", json=payload).status_code == 200
    assert client.post("/chat/text", json=payload).status_code == 200

    limited = client.post("/chat/text", json=payload)
    assert limited.status_code == 429
    assert limited.headers.get("Retry-After") == "60"


def test_rate_limit_is_per_session(client, monkeypatch):
    _stub_generate(monkeypatch, "I hear you.")
    monkeypatch.setattr(main_module, "RATE_LIMIT_ITEM", RateLimitItemPerMinute(1))

    first = f"test-{uuid.uuid4().hex}"
    second = f"test-{uuid.uuid4().hex}"
    assert client.post("/chat/text", json={"session_id": first, "message": "Hi.", "history": []}).status_code == 200
    assert client.post("/chat/text", json={"session_id": first, "message": "Hi.", "history": []}).status_code == 429
    # A different session must not inherit the exhausted budget of the first one.
    assert client.post("/chat/text", json={"session_id": second, "message": "Hi.", "history": []}).status_code == 200


# --- voice chat ----------------------------------------------------------


def test_voice_round_trip_preserves_newlines_and_unicode(client, monkeypatch, session_id):
    """Regression: these values used to ride in HTTP headers, which cannot hold them."""

    _stub_generate(monkeypatch, MULTILINE_REPLY)
    _stub_voice(monkeypatch)

    response = client.post(
        "/chat/voice",
        files={"audio_file": ("clip.webm", b"fake-audio-bytes", "audio/webm")},
        data={"session_id": session_id},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["transcript"] == UNICODE_TRANSCRIPT
    assert body["reply"] == MULTILINE_REPLY
    assert "\n" in body["reply"]
    assert body["session_id"] == session_id
    assert body["audio_media_type"] == "audio/wav"
    assert base64.b64decode(body["audio_base64"]) == BINARY_AUDIO


def test_voice_round_trip_flags_crisis(client, monkeypatch, session_id):
    _stub_generate(monkeypatch, "I am here with you.")
    _stub_voice(monkeypatch, transcript="I want to die.")

    response = client.post(
        "/chat/voice",
        files={"audio_file": ("clip.webm", b"fake-audio-bytes", "audio/webm")},
        data={"session_id": session_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crisis_detected"] is True
    assert CRISIS_RESOURCE_MESSAGE in body["reply"]


# --- history -------------------------------------------------------------


def test_history_round_trip_and_delete(client, monkeypatch, session_id):
    _stub_generate(monkeypatch, "I hear you.")
    client.post(
        "/chat/text",
        json={"session_id": session_id, "message": "I feel tired.", "history": []},
    )

    history = client.get(f"/chat/history/{session_id}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "I feel tired."

    deleted = client.delete(f"/chat/history/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == 2
    assert client.get(f"/chat/history/{session_id}").json()["messages"] == []
