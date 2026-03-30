"""End-to-end benchmark script for the Serenity local voice pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for voice pipeline testing."""

    parser = argparse.ArgumentParser(description="Benchmark Serenity voice round-trip latency.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to a WAV or WebM audio file.")
    parser.add_argument("--stt-url", default="http://127.0.0.1:8002/transcribe")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8003/speak")
    parser.add_argument("--prompt-prefix", default="User said: ")
    return parser


def main() -> int:
    """Run the voice pipeline benchmark against the local services."""

    parser = build_parser()
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    with httpx.Client(timeout=240.0) as client:
        started = time.perf_counter()
        with args.audio.open("rb") as handle:
            stt_response = client.post(
                args.stt_url,
                files={"audio_file": (args.audio.name, handle, "audio/wav")},
            )
        stt_response.raise_for_status()
        transcript = str(stt_response.json().get("transcript", "")).strip()
        stt_elapsed = time.perf_counter() - started

        llm_started = time.perf_counter()
        llm_response = client.post(
            args.llm_url,
            json={
                "prompt": f"{args.prompt_prefix}{transcript}",
                "max_tokens": 128,
                "temperature": 0.7,
            },
        )
        llm_response.raise_for_status()
        reply = str(llm_response.json().get("response", "")).strip()
        llm_elapsed = time.perf_counter() - llm_started

        tts_started = time.perf_counter()
        tts_response = client.post(args.tts_url, json={"text": reply})
        tts_response.raise_for_status()
        tts_elapsed = time.perf_counter() - tts_started

    report = {
        "transcript": transcript,
        "reply": reply,
        "latency_seconds": {
            "stt": round(stt_elapsed, 3),
            "llm": round(llm_elapsed, 3),
            "tts": round(tts_elapsed, 3),
            "total": round(stt_elapsed + llm_elapsed + tts_elapsed, 3),
        },
        "tts_bytes": len(tts_response.content),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
