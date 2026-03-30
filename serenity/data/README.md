# Serenity Data Pipeline

This directory contains the dataset preparation layer for Serenity's local mental health support model.

## Sources

- Counsel Chat
- EmpatheticDialogues
- PsyQA

## Output Contract

The merged dataset is exported as JSONL with one record per line:

```json
{"prompt": "...", "response": "..."}
```

## Cleaning Steps

- Redacts common PII patterns such as emails, phone numbers, SSNs, and URLs
- Normalizes whitespace and repeated newlines
- Deduplicates identical prompt-response pairs

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
