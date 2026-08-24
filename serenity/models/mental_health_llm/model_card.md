# Serenity Mental Health LLM

## Overview
Serenity is a locally fine-tuned assistant designed for compassionate, non-diagnostic mental health support. It listens empathetically, avoids clinical diagnosis, and includes crisis escalation guidance.

## Base Model
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Strategy: `qlora`
- Device target: `cuda`

## Training Data
- counsel_chat: 2612 examples
- empathetic_dialogues: 20000 examples
- Merged JSONL at `/app/serenity/data/processed/mental_health_dataset.jsonl`

## Training Parameters
- Epochs: `1`
- Learning rate: `0.0002`
- Batch size (requested): `4`
- Per-device batch: `4`
- Gradient accumulation: `4`
- LoRA rank: `16`
- LoRA alpha: `32`

## Safety Notes
- The model is for supportive conversation only and must not be used for diagnosis.
- This model does not answer crisis disclosures. When the detector in
  `backend/crisis_detector.py` fires, `backend/main.py` skips generation entirely
  and returns only the vetted resource message below, so nothing the model
  produces reaches a user in that moment.
- Crisis resource text returned in place of generation: `I'm really concerned about what you've shared. Please reach out to iCall India at 9152987821 or Vandrevala Foundation at 1860-2662-345 immediately. You are not alone.`
