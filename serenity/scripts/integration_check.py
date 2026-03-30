"""Basic HTTP integration checks for a running Serenity stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for backend smoke tests."""

    parser = argparse.ArgumentParser(description="Run Serenity integration checks against active services.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--voice-file", type=Path, default=None)
    return parser


def main() -> int:
    """Run text and optional voice endpoint checks against a live deployment."""

    args = build_parser().parse_args()
    report: dict[str, object] = {
      "text_chat": "FAIL",
      "voice_chat": "SKIPPED",
    }

    with httpx.Client(timeout=120.0) as client:
        text_response = client.post(
            f"{args.backend_url}/chat/text",
            json={"session_id": "integration-check", "message": "I have been stressed lately.", "history": []},
            headers={"X-Session-Id": "integration-check"},
        )
        text_response.raise_for_status()
        reply = text_response.json().get("reply", "")
        report["text_chat"] = "PASS" if reply else "FAIL"
        report["text_reply_preview"] = str(reply)[:120]

        if args.voice_file is not None and args.voice_file.exists():
            with args.voice_file.open("rb") as handle:
                voice_response = client.post(
                    f"{args.backend_url}/chat/voice",
                    files={"audio_file": (args.voice_file.name, handle, "audio/wav")},
                    data={"session_id": "integration-check-voice"},
                    headers={"X-Session-Id": "integration-check-voice"},
                )
            voice_response.raise_for_status()
            report["voice_chat"] = "PASS" if voice_response.content else "FAIL"
        elif args.voice_file is not None:
            report["voice_chat"] = f"FAIL (missing file: {args.voice_file})"

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
