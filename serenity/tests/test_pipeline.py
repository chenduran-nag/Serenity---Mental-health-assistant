"""Tests for corpus assembly and per-source failure handling.

data/pipeline.py imports `datasets` at module scope, which is a heavy dependency
belonging to requirements-data.txt rather than the test stack. These tests stub
it so corpus assembly can be exercised without installing the training stack.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

if "datasets" not in sys.modules:  # pragma: no cover - only runs without the real package
    _stub = types.ModuleType("datasets")
    for _name in ("Dataset", "DatasetDict", "IterableDataset"):
        setattr(_stub, _name, type(_name, (), {}))
    _stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = _stub

from data import pipeline  # noqa: E402
from data.pipeline import DataPrepConfig, build_training_corpus  # noqa: E402


COUNSEL_ROWS = [
    {"id": "c1", "questionText": "I feel anxious all the time.", "answerText": "That sounds exhausting."},
    {"id": "c2", "questionText": "I cannot sleep.", "answerText": "Let us look at your evenings."},
]

EMPATHETIC_ROWS = [
    {"conv_id": "e1", "speaker_idx": 0, "utterance": "I lost my job today."},
    {"conv_id": "e1", "speaker_idx": 1, "utterance": "I am sorry, that is a real blow."},
]

PSYQA_ROWS = [
    {"id": "p1", "question": "How do I cope with panic?", "answer": "Start with slow breathing."},
]


@pytest.fixture()
def config(tmp_path):
    return DataPrepConfig(output_path=tmp_path / "corpus.jsonl", cache_dir=tmp_path / "cache")


def _loader(mapping):
    """Build a load_source stub; a mapped value that is an Exception is raised."""

    def _load(source, cache_dir):
        outcome = mapping[source.name]
        if isinstance(outcome, Exception):
            raise outcome
        return [outcome]

    return _load


def test_revision_is_forwarded_to_load_dataset(monkeypatch, config):
    """EmpatheticDialogues only loads from the hub's auto-converted parquet revision."""

    calls = []

    def _fake_load_dataset(candidate, name=None, cache_dir=None, revision=None):
        calls.append((candidate, revision))
        raise RuntimeError("stop after recording the call")

    monkeypatch.setattr(pipeline, "load_dataset", _fake_load_dataset)

    with pytest.raises(RuntimeError):
        pipeline.load_source(config.empathetic_dialogues, config.cache_dir)

    assert calls, "load_dataset was never called"
    assert calls[0] == ("facebook/empathetic_dialogues", "refs/convert/parquet")
    # The bare id is still tried as a fallback.
    assert "empathetic_dialogues" in [candidate for candidate, _ in calls]


def test_all_sources_contribute(monkeypatch, config):
    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": COUNSEL_ROWS,
        "empathetic_dialogues": EMPATHETIC_ROWS,
        "psyqa": PSYQA_ROWS,
    }))

    result = build_training_corpus(config)

    assert result.skipped == []
    assert {outcome.name for outcome in result.outcomes if outcome.ok} == {
        "counsel_chat",
        "empathetic_dialogues",
        "psyqa",
    }
    assert result.examples


def test_gated_source_is_skipped_not_fatal(monkeypatch, config):
    """PsyQA is gated and 401s for most users; that must not sink the whole build."""

    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": COUNSEL_ROWS,
        "empathetic_dialogues": EMPATHETIC_ROWS,
        "psyqa": RuntimeError("Unable to load dataset 'psyqa'."),
    }))

    result = build_training_corpus(config)

    assert [outcome.name for outcome in result.skipped] == ["psyqa"]
    assert result.skipped[0].error is not None
    assert "psyqa" in result.skipped[0].error
    # The two working sources still produced a usable corpus.
    assert result.examples
    assert {outcome.name for outcome in result.outcomes if outcome.ok} == {
        "counsel_chat",
        "empathetic_dialogues",
    }


def test_total_failure_raises(monkeypatch, config):
    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": RuntimeError("counsel chat unavailable"),
        "empathetic_dialogues": RuntimeError("empathetic dialogues unavailable"),
        "psyqa": RuntimeError("psyqa unavailable"),
    }))

    with pytest.raises(RuntimeError, match="No datasets could be loaded"):
        build_training_corpus(config)


def test_sources_that_yield_nothing_raise(monkeypatch, config):
    """Loading successfully but extracting nothing is still an unusable corpus."""

    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": [],
        "empathetic_dialogues": [],
        "psyqa": [],
    }))

    with pytest.raises(RuntimeError, match="no usable prompt-response pairs"):
        build_training_corpus(config)


def test_examples_are_deduplicated(monkeypatch, config):
    duplicated = COUNSEL_ROWS + COUNSEL_ROWS

    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": duplicated,
        "empathetic_dialogues": RuntimeError("skip"),
        "psyqa": RuntimeError("skip"),
    }))

    result = build_training_corpus(config)

    pairs = [(example.prompt, example.response) for example in result.examples]
    assert len(pairs) == len(set(pairs))


def test_pii_is_redacted_but_ordinary_numbers_survive(monkeypatch, config):
    rows = [
        {"id": "x1", "question": "Call me at 555 123 4567.", "answer": "I have been on 50 mg for 3 years."},
    ]
    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": rows,
        "empathetic_dialogues": RuntimeError("skip"),
        "psyqa": RuntimeError("skip"),
    }))

    result = build_training_corpus(config)
    example = result.examples[0]

    assert "[redacted_phone]" in example.prompt
    assert "50 mg for 3 years" in example.response


def test_write_jsonl_emits_prompt_response_only(monkeypatch, config, tmp_path):
    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": COUNSEL_ROWS,
        "empathetic_dialogues": RuntimeError("skip"),
        "psyqa": RuntimeError("skip"),
    }))

    result = build_training_corpus(config)
    out = tmp_path / "corpus.jsonl"
    pipeline.write_jsonl(result.examples, out)

    import json

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines
    for record in lines:
        assert set(record) == {"prompt", "response"}


def test_summary_counts_by_source(monkeypatch, config):
    monkeypatch.setattr(pipeline, "load_source", _loader({
        "counsel_chat": COUNSEL_ROWS,
        "empathetic_dialogues": EMPATHETIC_ROWS,
        "psyqa": RuntimeError("skip"),
    }))

    result = build_training_corpus(config)
    summary = pipeline.dataset_summary(result.examples)

    assert summary["total_examples"] == len(result.examples)
    assert "psyqa" not in summary["by_source"]
    assert Path(config.output_path).name == "corpus.jsonl"
