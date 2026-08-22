"""Unit tests for the crisis detection safety layer."""

from __future__ import annotations

import pytest

from backend.crisis_detector import CRISIS_RESOURCE_MESSAGE, detect_crisis


CRISIS_PHRASES = [
    "I have been thinking about suicide.",
    "I feel suicidal today.",
    "I have been doing some self harm.",
    "I want to self-harm again.",
    "Sometimes I want to hurt myself.",
    "I think I want to end my life.",
    "I am going to kill myself.",
    "I just want to die.",
    "I cannot go on like this.",
    "I can't go on anymore.",
    "I thought about an overdose.",
    "I keep overdosing on my meds.",
]

BENIGN_PHRASES = [
    "I had a stressful week at work.",
    "My manager keeps ignoring me and it hurts.",
    "I am dyeing my hair to feel better.",
    "The presentation went badly and I feel embarrassed.",
    "I want to dye my hair blue.",
    "I have been feeling low but I am managing.",
]


@pytest.mark.parametrize("text", CRISIS_PHRASES)
def test_crisis_phrases_are_detected(text):
    result = detect_crisis(text)
    assert result.crisis_detected is True
    assert result.resource_message == CRISIS_RESOURCE_MESSAGE
    assert result.matched_patterns


@pytest.mark.parametrize("text", BENIGN_PHRASES)
def test_benign_phrases_are_not_flagged(text):
    result = detect_crisis(text)
    assert result.crisis_detected is False
    assert result.resource_message is None
    assert result.matched_patterns == ()


@pytest.mark.parametrize(
    "apostrophe",
    ["'", "’", "‘", "ʼ", "′"],
    ids=["ascii", "right-single-quote", "left-single-quote", "modifier", "prime"],
)
def test_typographic_apostrophes_are_normalized(apostrophe):
    """Phones and word processors substitute smart quotes; detection must survive it."""

    assert detect_crisis(f"I can{apostrophe}t go on").crisis_detected is True


def test_detection_is_case_insensitive():
    assert detect_crisis("I WANT TO DIE").crisis_detected is True
    assert detect_crisis("i WaNt To DiE").crisis_detected is True


def test_resource_message_names_both_helplines():
    assert "9152987821" in CRISIS_RESOURCE_MESSAGE
    assert "1860-2662-345" in CRISIS_RESOURCE_MESSAGE


def test_empty_input_is_safe():
    result = detect_crisis("")
    assert result.crisis_detected is False
