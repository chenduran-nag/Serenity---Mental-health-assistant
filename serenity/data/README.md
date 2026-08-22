# Serenity Data Pipeline

This directory contains the dataset preparation layer for Serenity's local mental health support model.

## Sources

| Source | Availability |
|---|---|
| Counsel Chat | public (`nbertagnolli/counsel-chat`, falls back to `Amod/mental_health_counseling_conversations`) |
| EmpatheticDialogues | public (`empathetic_dialogues`) |
| PsyQA | **gated** — not publicly downloadable; both candidate IDs return HTTP 401 |

Each source is loaded independently. One that fails is logged and skipped, and
recorded in `dataset_summary.json` under `sources` with the error that caused it.
The run fails only when *every* source fails, or when the merged corpus is empty.

In practice this means a default run builds from Counsel Chat and
EmpatheticDialogues. Supply `--psyqa-local` if you have been granted PsyQA access.

## Output Contract

The merged dataset is exported as JSONL with one record per line:

```json
{"prompt": "...", "response": "..."}
```

## Cleaning Steps

- Redacts common PII patterns such as emails, phone numbers, SSNs, and URLs
- Normalizes whitespace and repeated newlines
- Deduplicates identical prompt-response pairs

Redaction substitutes visible placeholders (`[redacted_phone]`, `[redacted_email]`).
A model fine-tuned on this corpus can therefore learn to emit those tokens
literally; dropping affected examples instead would avoid that.

## Running

```powershell
python scripts/download_data.py --output data/processed/mental_health_dataset.jsonl
```

Optional overrides are available if a dataset id differs in your environment:

```powershell
python scripts/download_data.py `
  --counsel-chat-local D:\datasets\counsel_chat.json `
  --psyqa-local D:\datasets\psyqa
```

The script prefers local paths when provided and otherwise attempts to load Hugging Face datasets using common candidate ids.
