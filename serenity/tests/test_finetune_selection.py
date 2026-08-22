"""Tests for hardware-based model selection in scripts/finetune.py.

scripts/finetune.py imports torch, datasets, and transformers at module scope.
Those belong to requirements-data.txt rather than the test stack, so they are
stubbed here; only the pure selection logic is under test.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_stubs():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda i: types.SimpleNamespace(total_memory=0),
            get_device_capability=lambda i: (0, 0),
        )
        torch.no_grad = lambda: None
        sys.modules["torch"] = torch

    if "datasets" not in sys.modules:
        ds = types.ModuleType("datasets")
        ds.Dataset = type("Dataset", (), {})
        ds.DatasetDict = type("DatasetDict", (), {})
        ds.IterableDataset = type("IterableDataset", (), {})
        ds.load_dataset = lambda *a, **k: None
        sys.modules["datasets"] = ds

    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        for name in (
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "BitsAndBytesConfig",
            "DataCollatorForLanguageModeling",
            "Trainer",
            "TrainingArguments",
        ):
            setattr(tf, name, type(name, (), {}))
        sys.modules["transformers"] = tf


_install_stubs()

from scripts import finetune  # noqa: E402


def _set_gpu(monkeypatch, total_bytes, capability=(7, 5), available=True):
    torch = sys.modules["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: types.SimpleNamespace(total_memory=total_bytes),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: capability, raising=False)


# Real values reported by CUDA, which sit below the nominal capacity.
RTX_2060_6GB = 6442123264      # 5.9997 GiB
RTX_3060_12GB = 12884901888    # 12.0 GiB nominal, reports ~11.99
RTX_3090_24GB = 25429442560    # ~23.68 GiB


def test_cpu_when_no_gpu(monkeypatch):
    _set_gpu(monkeypatch, 0, available=False)
    selection = finetune.detect_model_selection()
    assert selection.model_id == "distilgpt2"
    assert selection.device == "cpu"


def test_six_gb_card_uses_the_gpu(monkeypatch):
    """Regression: an RTX 2060 reports 5.9997 GiB, so a >= 6 test sent it to CPU."""

    _set_gpu(monkeypatch, RTX_2060_6GB)
    selection = finetune.detect_model_selection()
    assert selection.device == "cuda"
    assert selection.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert selection.strategy == "qlora"
    assert selection.load_in_4bit is True
    # 6 GB cannot hold a batch of 4 at this sequence length.
    assert selection.max_per_device_batch == 1
    assert selection.gradient_checkpointing is True


def test_twenty_four_gb_card_reaches_the_mistral_tier(monkeypatch):
    """Regression: a 24 GB card reports ~23.7 GiB, so a >= 24 test never matched."""

    _set_gpu(monkeypatch, RTX_3090_24GB)
    selection = finetune.detect_model_selection()
    assert selection.model_id == "mistralai/Mistral-7B-Instruct-v0.2"
    assert selection.strategy == "qlora"
    assert selection.device == "cuda"


def test_twelve_gb_card_gets_a_larger_batch(monkeypatch):
    _set_gpu(monkeypatch, RTX_3060_12GB)
    selection = finetune.detect_model_selection()
    assert selection.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert selection.max_per_device_batch == 4


def test_no_gpu_below_compute_capability_7_5(monkeypatch):
    """bitsandbytes NF4 needs 7.5+; older cards must not be handed a 4-bit path."""

    _set_gpu(monkeypatch, RTX_3090_24GB, capability=(6, 1))
    selection = finetune.detect_model_selection()
    assert selection.device == "cpu"
    assert selection.load_in_4bit is False


@pytest.mark.parametrize(
    "total_bytes",
    [RTX_2060_6GB, RTX_3060_12GB, RTX_3090_24GB],
    ids=["6gb", "12gb", "24gb"],
)
def test_every_supported_gpu_uses_qlora_not_full_finetune(monkeypatch, total_bytes):
    """Full fine-tuning even a 1.1B model needs ~18 GB once AdamW states count."""

    _set_gpu(monkeypatch, total_bytes)
    selection = finetune.detect_model_selection()
    assert selection.strategy == "qlora"
    assert selection.use_lora is True
