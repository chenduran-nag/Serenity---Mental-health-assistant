# Serenity Mental Health LLM

This directory is reserved for the locally fine-tuned Serenity assistant model.

## Intended base models

- `mistralai/Mistral-7B-Instruct-v0.2` with QLoRA when VRAM is at least 24 GB
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with full fine-tuning when VRAM is at least 12 GB
- `distilgpt2` for CPU-only fallback

## Training corpus

- Counsel Chat
- EmpatheticDialogues
- PsyQA

## Default training parameters

- Learning rate: `2e-4`
- Epochs: `3`
- Batch size: `4`
- LoRA rank: `16`
- LoRA alpha: `32`

`scripts/finetune.py` will replace this placeholder with run-specific metadata and save the trained artifacts in this directory.
