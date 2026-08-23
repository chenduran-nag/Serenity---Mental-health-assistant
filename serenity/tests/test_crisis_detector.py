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

# Inflected forms. Every one of these was missed by the original fixed-string
# patterns; the first was observed going undetected against the live stack.
INFLECTED_CRISIS_PHRASES = [
    "I have been thinking about ending my life.",
    "I keep thinking about killing myself.",
    "I have been hurting myself again.",
    "I have been harming myself.",
    "I have been cutting myself.",
    "I have been self-harming for months.",
    "I wanted to die last night.",
    "I have been wanting to die.",
    "I am thinking of taking my own life.",
    "I might just end it all.",
    "I don't want to live anymore.",
    "I don't want to wake up tomorrow.",
    "Everyone would be better off dead without me.",
    "There is no reason to live.",
    "I see no point in living.",
    "I have thought about ending my own life.",
]


@pytest.mark.parametrize("text", INFLECTED_CRISIS_PHRASES)
def test_inflected_crisis_phrases_are_detected(text):
    """Regression: 'ending my life' went undetected because only 'end my life' matched."""

    result = detect_crisis(text)
    assert result.crisis_detected is True, f"missed crisis disclosure: {text!r}"
    assert result.resource_message == CRISIS_RESOURCE_MESSAGE

BENIGN_PHRASES = [
    "I had a stressful week at work.",
    "My manager keeps ignoring me and it hurts.",
    "I am dyeing my hair to feel better.",
    "The presentation went badly and I feel embarrassed.",
    "I want to dye my hair blue.",
    "I have been feeling low but I am managing.",
    # Idioms that sit close to the crisis vocabulary without being disclosures.
    "My phone died during the call.",
    "The deadline is killing me at work.",
    "I ended my subscription yesterday.",
    "That movie was to die for.",
    "I am dead tired after that shift.",
    "I cut myself a slice of cake.",
    "I cut myself some slack for once.",
    "I want to live abroad someday.",
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
