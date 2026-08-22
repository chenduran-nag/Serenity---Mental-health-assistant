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

> **VAD is running but not yet wired to the UI.** `backend/vad_server.py` serves
> `ws://.../vad` and is covered by the health probe, but `VoiceInterface.jsx`
> currently uses manual tap-to-start / tap-to-stop recording and never opens that
> socket. Automatic end-of-speech detection is still to be built on top of it.

## Project Layout

```text
serenity/
├── backend/
├── data/
├── frontend/
├── models/
├── scripts/
├── tests/
├── docker-compose.yml
├── requirements.txt          # shared API layer
├── requirements-*.txt        # per-service dependency stacks
├── start.sh
└── README.md
```

## Setup

> **Serenity needs a fine-tuned model before it can reply.** `models/mental_health_llm/`
> ships with only a placeholder model card. Until you run the data prep and
> fine-tuning steps below, the model server starts but reports itself unhealthy,
> and `/chat/text` returns 503. Nothing in `./start.sh` or `docker-compose.yml`
> trains a model for you.

> **Python 3.11 specifically.** Coqui TTS declares `>=3.9,<3.12`, which caps the
> whole project. The Docker images use `python:3.11-slim` for this reason.

### Dependencies

Each service installs only its own stack, so the Whisper, Coqui, and training
pins never have to resolve against one another:

| File | Used by |
|---|---|
| `requirements.txt` | shared API layer, and `backend/main.py` |
| `requirements-model.txt` | `backend/model_server.py` |
| `requirements-stt.txt` | `backend/stt_server.py` |
| `requirements-tts.txt` | `backend/tts_server.py` |
| `requirements-vad.txt` | `backend/vad_server.py` |
| `requirements-data.txt` | `scripts/download_data.py`, `scripts/finetune.py` |
| `requirements-dev.txt` | the test suite |

### Docker path

1. From `serenity/`, run `chmod +x start.sh` if needed.
2. Prepare data and train a model (see [Fine-Tuning](#fine-tuning)) — the stack
   starts without this, but cannot answer.
3. Start the full stack with `./start.sh`.
4. Open `http://localhost:5173`.
5. Check `http://localhost:8000/health` to see which services came up.

Compose builds a separate image per service, each with its own `REQUIREMENTS`
build argument. `depends_on` controls start order only; it deliberately does not
gate on health, since an untrained model server would otherwise block the whole
stack from starting.

#### Disk usage and CPU vs GPU wheels

torch is installed in a dedicated layer by a command identical across every
service, so the image store keeps **one** copy instead of one per service.

That layer uses the **CPU wheel index by default**. The default compose config
sets `NVIDIA_VISIBLE_DEVICES=none`, so the CUDA runtime bundled into the PyPI
wheels would be roughly 3 GB of dead weight per image. Building all services with
the default PyPI wheels needs well over 20 GB of Docker disk.

To build GPU images, point at the matching CUDA index:

```bash
SERENITY_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 docker compose build
```

If Docker runs out of space mid-build, the symptom is misleading: the VM's
filesystem remounts read-only and pip reports
`OSError: [Errno 30] Read-only file system` rather than a disk-space error.

### Non-Docker path

1. Create a Python 3.11 environment.
2. Prepare data and fine-tune:
   - `pip install -r requirements-data.txt`
   - `python scripts/download_data.py`
   - `python scripts/finetune.py`
3. Install per-service dependencies in separate environments, then run each
   service in its own terminal:
   - `pip install -r requirements-model.txt` → `uvicorn backend.model_server:app --host 0.0.0.0 --port 8001`
   - `pip install -r requirements-stt.txt` → `uvicorn backend.stt_server:app --host 0.0.0.0 --port 8002`
   - `pip install -r requirements-tts.txt` → `uvicorn backend.tts_server:app --host 0.0.0.0 --port 8003`
   - `pip install -r requirements-vad.txt` → `uvicorn backend.vad_server:app --host 0.0.0.0 --port 8004`
   - `pip install -r requirements.txt` → `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
4. Install frontend dependencies with `cd frontend && npm install`, then `npm run dev`.

`openai-whisper` and `webrtcvad` are source-only distributions, so a working C
toolchain is needed outside Docker. The images install `build-essential` for this.

## Fine-Tuning

### Data preparation

- `scripts/download_data.py` merges Counsel Chat, EmpatheticDialogues, and PsyQA into `data/processed/mental_health_dataset.jsonl`
- Cleaning includes whitespace normalization, PII redaction, and deduplication
- Sources are independent. A source that cannot be loaded is logged, skipped, and
  recorded in `dataset_summary.json` under `sources`; the run fails only if every
  source fails, or if the merged corpus ends up empty.

> **PsyQA is gated.** It is not publicly downloadable from the Hugging Face hub —
> both candidate IDs return HTTP 401 — so by default the corpus is built from
> Counsel Chat and EmpatheticDialogues alone. If you have been granted access,
> point at your local copy:
>
> ```bash
> python scripts/download_data.py --psyqa-local /path/to/psyqa
> ```

Note that the PII redaction substitutes visible placeholders such as
`[redacted_phone]`. A model fine-tuned on this corpus can learn to emit those
tokens verbatim; dropping affected examples instead is worth considering.

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
- Returns `application/json`

```json
{
  "reply": "empathetic reply",
  "transcript": "what the user said",
  "session_id": "uuid",
  "timestamp": "2026-03-29T00:00:00+00:00",
  "crisis_detected": false,
  "crisis_message": null,
  "audio_base64": "UklGRiQAAABXQVZF...",
  "audio_media_type": "audio/wav"
}
```

Audio is base64-encoded inside the JSON body rather than returned as a raw
`audio/wav` response with metadata in `X-` headers. HTTP header values cannot
contain newlines and must be latin-1 encodable. Model replies routinely contain
newlines — the crisis path always does — and transcripts can be in any script,
so the header-based contract could not carry either one reliably.

### `GET /chat/history/{session_id}`

- Returns persisted message history from SQLite

### `DELETE /chat/history/{session_id}`

- Deletes the session history

### `GET /health`

- Returns readiness for the model, STT, TTS, and VAD services
- Responds `503` when any of them is unavailable, with a per-service `error`
- A service that failed to load its model stays up and reports the reason here,
  rather than crash-looping

## Configuration

All services read plain environment variables; the defaults target a local run.

| Variable | Default | Used by |
|---|---|---|
| `SERENITY_MODEL_SERVER_URL` | `http://127.0.0.1:8001` | backend |
| `SERENITY_STT_SERVER_URL` | `http://127.0.0.1:8002` | backend |
| `SERENITY_TTS_SERVER_URL` | `http://127.0.0.1:8003` | backend |
| `SERENITY_VAD_SERVER_URL` | `http://127.0.0.1:8004` | backend |
| `SERENITY_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | backend |
| `SERENITY_RATE_LIMIT_PER_MINUTE` | `60` | backend |
| `SERENITY_HTTP_TIMEOUT` | `120` | backend |
| `SERENITY_DATABASE_URL` | `sqlite:///<repo>/serenity.db` | backend |
| `SERENITY_MODEL_DIR` | `models/mental_health_llm` | model server |
| `WHISPER_MODEL_NAME` | chosen by available RAM | STT server |
| `TTS_MODEL_NAME` | `tts_models/en/ljspeech/glow-tts` | TTS server |

CORS is restricted to the listed origins rather than `*`, since a wildcard would
let any site the user visits post to their local backend.

## Safety Layer

Serenity always runs the crisis detector before returning chat output. Messages matching suicide or self-harm patterns are flagged and paired with the following crisis resource message:

> I'm really concerned about what you've shared. Please reach out to iCall India at 9152987821 or Vandrevala Foundation at 1860-2662-345 immediately. You are not alone.

The detector runs on the user's message *before* generation, and the resource
message is returned even if the model server fails, so an outage cannot suppress
it. Typographic apostrophes are normalized first, so `I can't go on` is caught
whether it was typed with `'` or the `’` that phones substitute automatically.

### Data handling

Conversations are written to SQLite in plaintext at `serenity.db`, which is
bind-mounted into the containers. There is no encryption at rest and no
retention limit. `DELETE /chat/history/{session_id}` is the only way to remove a
session. Consider this before running Serenity anywhere but your own machine.

## Hardware Guide

| Profile | Recommended hardware | Expected model path |
|---|---|---|
| CPU-only | 16 GB RAM | `distilgpt2` |
| Mid GPU | 12 GB VRAM + 16 GB RAM | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| High GPU | 24 GB VRAM + 32 GB RAM | `mistralai/Mistral-7B-Instruct-v0.2` with QLoRA |

## Tests

The unit and API tests stub the model, STT, and TTS services, so they need only
the light dependencies — no torch, Whisper, or Coqui install required.

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

`tests/test_crisis_detector.py` covers the safety layer in isolation.
`tests/test_main.py` covers the orchestration endpoints, including regression
tests for the per-session rate limiter, for crisis resources surviving a model
server outage, and for voice replies that contain newlines or non-latin-1 text.

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
