# Serenity

Serenity is a fully local AI mental health assistant with text and voice interfaces. It prepares mental health conversation datasets, fine-tunes a local language model, serves that model through FastAPI, and pairs it with a React frontend shaped like a bioluminescent deep-ocean chamber.

## Architecture

```text
                          +----------------------+
                          |  React + Vite UI     |
                          |  Text / Voice Modes  |
                          +----------+-----------+
                                     |
                                     v
                         +-----------+------------+
                         |    FastAPI Backend     |
                         |  sessions + safety     |
                         +---+---------+------+---+
                             |         |      |
                +------------+         |      +------------------+
                v                      v                         v
      +---------+----------+  +--------+---------+   +-----------+-----------+
      |  Model Server       |  |  STT Server      |   |  TTS / VAD Services   |
      |  local fine-tuned   |  |  local Whisper   |   |  Coqui + webrtcvad    |
      +---------+----------+  +-------------------+   +-----------------------+
                |
                v
      +---------+----------+
      | Fine-tuned model   |
      | ./models/...       |
      +--------------------+

      Data Prep -> Counsel Chat + EmpatheticDialogues + PsyQA -> merged JSONL
```

## Project Layout

```text
serenity/
├── backend/
├── data/
├── frontend/
├── models/
├── scripts/
├── docker-compose.yml
├── requirements.txt
├── start.sh
└── README.md
```

## Setup

### Docker path

1. From `serenity/`, run `chmod +x start.sh` if needed.
2. Start the full stack with `./start.sh`.
3. Open `http://localhost:5173`.

### Non-Docker path

1. Create a Python 3.11 environment.
2. Install backend dependencies with `pip install -r requirements.txt`.
3. Install frontend dependencies with `cd frontend && npm install`.
4. Prepare data:
   - `python scripts/download_data.py`
5. Fine-tune the model:
   - `python scripts/finetune.py`
6. Run services in separate terminals:
   - `uvicorn backend.model_server:app --host 0.0.0.0 --port 8001`
   - `uvicorn backend.stt_server:app --host 0.0.0.0 --port 8002`
   - `uvicorn backend.tts_server:app --host 0.0.0.0 --port 8003`
   - `uvicorn backend.vad_server:app --host 0.0.0.0 --port 8004`
   - `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
   - `cd frontend && npm run dev`

## Fine-Tuning

### Data preparation

- `scripts/download_data.py` merges Counsel Chat, EmpatheticDialogues, and PsyQA into `data/processed/mental_health_dataset.jsonl`
- Cleaning includes whitespace normalization, PII redaction, and deduplication

### Model selection logic

- `>= 24 GB VRAM`: `mistralai/Mistral-7B-Instruct-v0.2` with QLoRA
- `>= 12 GB VRAM`: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` full fine-tune
- CPU only: `distilgpt2` full fine-tune

### Training defaults

- Learning rate: `2e-4`
- Epochs: `3`
- Batch size: `4`
- LoRA rank: `16`
- LoRA alpha: `32`

Artifacts are saved under `models/mental_health_llm/`, including `training_metadata.json` and `model_card.md`.

## API

### `POST /chat/text`

Request:

```json
{
  "session_id": "uuid-or-null",
  "message": "I've been overwhelmed today.",
  "history": [
    {
      "role": "user",
      "content": "I have had a hard week."
    }
  ]
}
```

Response:

```json
{
  "reply": "empathetic reply",
  "session_id": "uuid",
  "timestamp": "2026-03-29T00:00:00+00:00",
  "crisis_detected": false,
  "crisis_message": null
}
```

### `POST /chat/voice`

- Multipart field: `audio_file`
- Optional multipart field: `session_id`
- Returns `audio/wav`
- Response headers include:
  - `X-Transcript`
  - `X-Reply-Text`
  - `X-Session-Id`
  - `X-Crisis-Detected`

### `GET /chat/history/{session_id}`

- Returns persisted message history from SQLite

### `DELETE /chat/history/{session_id}`

- Deletes the session history

### `GET /health`

- Returns readiness for backend, model, STT, and TTS services

## Safety Layer

Serenity always runs the crisis detector before returning chat output. Messages matching suicide or self-harm patterns are flagged and paired with the following crisis resource message:

> I'm really concerned about what you've shared. Please reach out to iCall India at 9152987821 or Vandrevala Foundation at 1860-2662-345 immediately. You are not alone.

## Hardware Guide

| Profile | Recommended hardware | Expected model path |
|---|---|---|
| CPU-only | 16 GB RAM | `distilgpt2` |
| Mid GPU | 12 GB VRAM + 16 GB RAM | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| High GPU | 24 GB VRAM + 32 GB RAM | `mistralai/Mistral-7B-Instruct-v0.2` with QLoRA |

## Integration Checks

### Text chat

```bash
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: smoke-test" \
  -d '{"session_id":"smoke-test","message":"I feel anxious today.","history":[]}'
```

### Voice chat

```bash
curl -X POST http://localhost:8000/chat/voice \
  -H "X-Session-Id: smoke-voice" \
  -F "session_id=smoke-voice" \
  -F "audio_file=@sample.wav"
```

You can also run `python scripts/integration_check.py --voice-file sample.wav` once the stack is up.
