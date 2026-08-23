"""Tests for model server prompt construction and completion trimming.

backend/model_server.py imports torch and transformers at module scope. Those
belong to requirements-model.txt rather than the test stack, so they are stubbed
here; only the pure text handling is under test.
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

    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        for name in ("AutoModelForCausalLM", "AutoTokenizer"):
            setattr(tf, name, type(name, (), {}))
        sys.modules["transformers"] = tf


_install_stubs()

from backend.model_server import ChatTurn, trim_to_single_turn, _format_prompt  # noqa: E402


# --- trimming -------------------------------------------------------------


def test_completion_running_into_the_next_turn_is_cut():
    """Observed on a real fine-tune: the model answers, then invents the user's reply."""

    raw = (
        "I can understand that. What is going on?\n"
        "[/ASSISTANT]\n[USER]\nI'm just trying to keep my head above water."
    )
    assert trim_to_single_turn(raw) == "I can understand that. What is going on?"


@pytest.mark.parametrize(
    "marker",
    ["[/ASSISTANT]", "[ASSISTANT]", "[USER]", "[/USER]", "[SYSTEM]", "<s>", "</s>"],
)
def test_every_turn_marker_terminates_the_completion(marker):
    assert trim_to_single_turn(f"That sounds hard. {marker} trailing junk") == "That sounds hard."


def test_clean_completion_is_left_alone():
    text = "That sounds genuinely difficult, and I am glad you told me."
    assert trim_to_single_turn(text) == text


def test_earliest_marker_wins():
    raw = "Short answer.[USER]\nsomething[/ASSISTANT]more"
    assert trim_to_single_turn(raw) == "Short answer."


def test_surrounding_whitespace_is_stripped():
    assert trim_to_single_turn("\n  Take a breath.  \n[/ASSISTANT]") == "Take a breath."


# --- prompt construction --------------------------------------------------


def test_prompt_without_history_matches_training_layout():
    prompt = _format_prompt("I feel anxious.")
    assert "[SYSTEM]" in prompt
    assert "[USER]\nI feel anxious.\n[/USER]" in prompt
    assert prompt.rstrip().endswith("[ASSISTANT]")


def test_history_is_rendered_as_separate_turn_blocks():
    history = [
        ChatTurn(role="user", content="I had a hard week."),
        ChatTurn(role="assistant", content="That sounds heavy."),
    ]
    prompt = _format_prompt("And today was worse.", history)

    assert "[USER]\nI had a hard week.\n[/USER]" in prompt
    assert "[ASSISTANT]\nThat sounds heavy.\n[/ASSISTANT]" in prompt
    # The latest message is last, with an open assistant block for the model.
    assert prompt.index("I had a hard week.") < prompt.index("And today was worse.")
    assert prompt.rstrip().endswith("[ASSISTANT]")


def test_blank_history_turns_are_skipped():
    history = [ChatTurn(role="user", content="   ")]
    prompt = _format_prompt("Hello.", history)
    assert prompt.count("[USER]") == 1
