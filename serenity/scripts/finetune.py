"""Fine-tune a local language model for the Serenity assistant."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.crisis_detector import CRISIS_RESOURCE_MESSAGE


SYSTEM_PROMPT = (
    "You are Serenity, a compassionate, non-judgmental mental health support assistant. "
    "You do not diagnose. You listen, validate, and guide with empathy. Always recommend "
    "professional help for serious concerns."
)


@dataclass(frozen=True)
class ModelSelection:
    """Selected training strategy based on available hardware."""

    model_id: str
    strategy: str
    load_in_4bit: bool
    use_lora: bool
    target_modules: tuple[str, ...]
    device: str
    max_per_device_batch: int
    gradient_checkpointing: bool


@dataclass(frozen=True)
class TrainingMetadata:
    """Persisted metadata alongside the trained model."""

    base_model: str
    strategy: str
    device: str
    epochs: int
    learning_rate: float
    batch_size: int
    lora_r: int
    lora_alpha: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    dataset_sources: dict[str, int]
    dataset_path: str
    model_output_dir: str
    crisis_resource_message: str


def detect_model_selection() -> ModelSelection:
    """Select an appropriate base model based on local hardware."""

    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # bitsandbytes NF4 needs compute capability 7.5 (Turing) or newer.
        supports_4bit = torch.cuda.get_device_capability(0) >= (7, 5)

        # Thresholds sit below nominal capacity on purpose. CUDA reports slightly
        # less memory than a card is sold with - a 6 GB RTX 2060 reports 5.9997
        # GiB, and a 24 GB RTX 3090 reports about 23.7 - so testing against the
        # nominal figure never matches the cards it is meant to describe.
        if supports_4bit and total_gb >= 22:
            return ModelSelection(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
                strategy="qlora",
                load_in_4bit=True,
                use_lora=True,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                device="cuda",
                max_per_device_batch=4,
                gradient_checkpointing=True,
            )
        if supports_4bit and total_gb >= 5.5:
            # Fully fine-tuning even a 1.1B model needs roughly 18 GB once AdamW
            # optimizer states are counted, so every GPU below 24 GB uses QLoRA.
            # Cards in the 6-8 GB range additionally need a per-device batch of 1,
            # with the effective batch made up by gradient accumulation.
            return ModelSelection(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                strategy="qlora",
                load_in_4bit=True,
                use_lora=True,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                device="cuda",
                # Measured on a 6 GB RTX 2060: a per-device batch of 1 used only
                # 1.9 GB and left the GPU 39% utilised while the CPU saturated, so
                # the small-card path was input-bound rather than memory-bound.
                max_per_device_batch=4,
                gradient_checkpointing=True,
            )

    return ModelSelection(
        model_id="distilgpt2",
        strategy="full_finetune",
        load_in_4bit=False,
        use_lora=False,
        target_modules=(),
        device="cpu",
        max_per_device_batch=4,
        gradient_checkpointing=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the fine-tuning workflow."""

    parser = argparse.ArgumentParser(description="Fine-tune Serenity's local LLM.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "processed" / "mental_health_dataset.jsonl",
        help="Path to the preprocessed JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "mental_health_llm",
        help="Directory for model artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Checkpoint interval in optimizer steps. Keeps a long run resumable.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory, or 'auto' to pick the latest in --output-dir.",
    )
    return parser


def load_training_dataset(dataset_path: Path) -> Dataset:
    """Load the merged JSONL training data."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {dataset_path}. Run scripts/download_data.py first."
        )

    dataset_dict = load_dataset("json", data_files=str(dataset_path))
    return dataset_dict["train"]


def build_example_text(prompt: str, response: str) -> str:
    """Construct a supervised fine-tuning prompt."""

    return (
        f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n[/SYSTEM]\n"
        f"[USER]\n{prompt.strip()}\n[/USER]\n"
        f"[ASSISTANT]\n{response.strip()}\n[/ASSISTANT]</s>"
    )


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    """Tokenize the dataset into causal language modeling sequences."""

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = build_example_text(example["prompt"], example["response"])
        # No padding here: DataCollatorForLanguageModeling pads each batch to its
        # own longest sequence and masks pad positions to -100. Padding every
        # example to max_length instead would burn compute on padding for the 87%
        # of this corpus that is far shorter, and would train on pad tokens.
        return tokenizer(text, truncation=True, max_length=max_length)

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def create_model_and_tokenizer(
    selection: ModelSelection,
    lora_r: int,
    lora_alpha: int,
) -> tuple[Any, AutoTokenizer]:
    """Instantiate the base model and tokenizer for the selected strategy."""

    tokenizer = AutoTokenizer.from_pretrained(selection.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if selection.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if selection.device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(selection.model_id, **model_kwargs)

    if selection.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(selection.target_modules),
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer


def _resolve_checkpoint(args: argparse.Namespace) -> str | bool | None:
    """Resolve --resume-from-checkpoint into what Trainer.train expects."""

    requested = args.resume_from_checkpoint
    if requested is None:
        return None
    if requested != "auto":
        return requested

    checkpoints = sorted(
        (path for path in args.output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        print("No checkpoint found; starting from scratch.")
        return None
    print(f"Resuming from {checkpoints[-1]}")
    return str(checkpoints[-1])


def run_training(args: argparse.Namespace) -> TrainingMetadata:
    """Execute the fine-tuning workflow and persist artifacts."""

    selection = detect_model_selection()
    dataset = load_training_dataset(args.dataset)
    model, tokenizer = create_model_and_tokenizer(selection, args.lora_r, args.lora_alpha)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_device_batch = max(1, min(args.batch_size, selection.max_per_device_batch))
    gradient_accumulation_steps = max(1, math.ceil(16 / per_device_batch))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=selection.gradient_checkpointing,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        # Step-based checkpoints: a machine that sleeps or reboots mid-run would
        # otherwise lose everything, since one epoch over this corpus is hours.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=2,
        report_to=[],
        fp16=selection.device == "cuda",
        bf16=False,
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if selection.load_in_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train(resume_from_checkpoint=_resolve_checkpoint(args))

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = TrainingMetadata(
        base_model=selection.model_id,
        strategy=selection.strategy,
        device=selection.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_sources=_dataset_sources(args.dataset),
        dataset_path=str(args.dataset),
        model_output_dir=str(args.output_dir),
        crisis_resource_message=CRISIS_RESOURCE_MESSAGE,
    )
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "model_card.md").write_text(build_model_card(metadata), encoding="utf-8")
    return metadata


def _dataset_sources(dataset_path: Path) -> dict[str, int]:
    """Read which sources actually contributed, written by download_data.py."""

    summary_path = dataset_path.parent / "dataset_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return dict(json.loads(summary_path.read_text(encoding="utf-8")).get("by_source", {}))
    except (ValueError, OSError):
        return {}


def build_model_card(metadata: TrainingMetadata) -> str:
    """Generate a concise model card markdown document."""

    # List the sources that actually contributed. The previous template hardcoded
    # all three, which credited PsyQA even though it is gated and always skipped.
    if metadata.dataset_sources:
        sources_block = "\n".join(
            f"- {name}: {count} examples"
            for name, count in sorted(metadata.dataset_sources.items())
        )
    else:
        sources_block = "- See `dataset_summary.json` beside the dataset."

    return f"""# Serenity Mental Health LLM

## Overview
Serenity is a locally fine-tuned assistant designed for compassionate, non-diagnostic mental health support. It listens empathetically, avoids clinical diagnosis, and includes crisis escalation guidance.

## Base Model
- `{metadata.base_model}`
- Strategy: `{metadata.strategy}`
- Device target: `{metadata.device}`

## Training Data
{sources_block}
- Merged JSONL at `{metadata.dataset_path}`

## Training Parameters
- Epochs: `{metadata.epochs}`
- Learning rate: `{metadata.learning_rate}`
- Batch size (requested): `{metadata.batch_size}`
- Per-device batch: `{metadata.per_device_batch_size}`
- Gradient accumulation: `{metadata.gradient_accumulation_steps}`
- LoRA rank: `{metadata.lora_r}`
- LoRA alpha: `{metadata.lora_alpha}`

## Safety Notes
- The model is for supportive conversation only and must not be used for diagnosis.
- This model does not answer crisis disclosures. When the detector in
  `backend/crisis_detector.py` fires, `backend/main.py` skips generation entirely
  and returns only the vetted resource message below, so nothing the model
  produces reaches a user in that moment.
- Crisis resource text returned in place of generation: `{metadata.crisis_resource_message}`
"""


def main() -> int:
    """CLI entrypoint for model fine-tuning."""

    parser = build_parser()
    args = parser.parse_args()
    metadata = run_training(args)
    print(json.dumps(asdict(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
