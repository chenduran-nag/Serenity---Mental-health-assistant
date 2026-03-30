"""Local model serving API for the Serenity fine-tuned LLM."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are Serenity, a compassionate, non-judgmental mental health support assistant. "
    "You do not diagnose. You listen, validate, and guide with empathy. Always recommend "
    "professional help for serious concerns."
)

MODEL_DIR = Path(os.getenv("SERENITY_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "mental_health_llm"))
MODEL_SERVER_PORT = int(os.getenv("SERENITY_MODEL_PORT", "8001"))
MAX_CONTEXT_LENGTH = int(os.getenv("SERENITY_MODEL_MAX_CONTEXT", "1024"))

app = FastAPI(title="Serenity Model Server", version="1.0.0")
tokenizer: Any | None = None
model: Any | None = None
metadata: dict[str, Any] = {}


class GenerateRequest(BaseModel):
    """Request payload for text generation."""

    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=256, ge=32, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """Response payload for text generation."""

    response: str


def _load_metadata() -> dict[str, Any]:
    """Load training metadata if available."""

    metadata_path = MODEL_DIR / "training_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def _load_model() -> tuple[Any, Any]:
    """Load the local model artifacts."""

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found at {MODEL_DIR}. Run scripts/finetune.py before starting the server."
        )

    tokenizer_obj = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token

    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if (MODEL_DIR / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM

        model_obj = AutoPeftModelForCausalLM.from_pretrained(str(MODEL_DIR), **model_kwargs)
    else:
        model_obj = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), **model_kwargs)

    model_obj.eval()
    return tokenizer_obj, model_obj


def _format_prompt(user_prompt: str) -> str:
    """Wrap the raw prompt with the required system behavior."""

    return (
        f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n[/SYSTEM]\n"
        f"[USER]\n{user_prompt.strip()}\n[/USER]\n"
        "[ASSISTANT]\n"
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model on service startup."""

    global tokenizer, model, metadata
    metadata = _load_metadata()
    tokenizer, model = _load_model()


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    """Generate a response from the locally fine-tuned model."""

    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    prompt = _format_prompt(payload.prompt)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=payload.max_tokens,
            temperature=payload.temperature,
            do_sample=payload.temperature > 0,
            top_p=0.92,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    if not generated:
        raise HTTPException(status_code=502, detail="Model generated an empty response.")
    return GenerateResponse(response=generated)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return model server health and metadata."""

    return {
        "ok": tokenizer is not None and model is not None,
        "model_dir": str(MODEL_DIR),
        "metadata": metadata,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.model_server:app", host="127.0.0.1", port=MODEL_SERVER_PORT, reload=False)
