"""Dataset preparation utilities for Serenity fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset


PROMPT_FIELDS: tuple[str, ...] = (
    "prompt",
    "question",
    "questionText",
    "query",
    "input",
    "instruction",
    "title",
    "context",
)

RESPONSE_FIELDS: tuple[str, ...] = (
    "response",
    "answer",
    "answerText",
    "completion",
    "output",
    "advice",
    "label",
    "utterance",
)

PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE), "[redacted_email]"),
    (
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)\d{3,4}[-.\s]?\d{3,4}\b"),
        "[redacted_phone]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted_ssn]"),
    (
        re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE),
        "[redacted_url]",
    ),
)


@dataclass(slots=True)
class DatasetSource:
    """Describes where a dataset should be loaded from."""

    name: str
    candidates: Sequence[str]
    split_candidates: Sequence[str]
    local_path: str | None = None
    subset: str | None = None


@dataclass(slots=True)
class DataPrepConfig:
    """Configuration for building the Serenity training corpus."""

    output_path: Path
    cache_dir: Path = Path("data/cache")
    counsel_chat: DatasetSource = field(
        default_factory=lambda: DatasetSource(
            name="counsel_chat",
            candidates=(
                "nbertagnolli/counsel-chat",
                "Amod/mental_health_counseling_conversations",
            ),
            split_candidates=("train",),
        )
    )
    empathetic_dialogues: DatasetSource = field(
        default_factory=lambda: DatasetSource(
            name="empathetic_dialogues",
            candidates=("empathetic_dialogues",),
            split_candidates=("train", "validation", "test"),
        )
    )
    psyqa: DatasetSource = field(
        default_factory=lambda: DatasetSource(
            name="psyqa",
            candidates=(
                "wangrongsheng/psyqa",
                "PsyQA",
            ),
            split_candidates=("train", "validation", "test"),
        )
    )


@dataclass(slots=True)
class PreparedExample:
    """Normalized training record."""

    prompt: str
    response: str
    source: str
    source_id: str

    def to_jsonl_record(self) -> dict[str, str]:
        """Return the export schema required for training."""

        return {
            "prompt": self.prompt,
            "response": self.response,
        }


def build_training_corpus(config: DataPrepConfig) -> list[PreparedExample]:
    """Load all configured datasets and return deduplicated examples."""

    all_examples: list[PreparedExample] = []
    all_examples.extend(_extract_counsel_chat(load_source(config.counsel_chat, config.cache_dir)))
    all_examples.extend(
        _extract_empathetic_dialogues(load_source(config.empathetic_dialogues, config.cache_dir))
    )
    all_examples.extend(_extract_psyqa(load_source(config.psyqa, config.cache_dir)))
    return deduplicate_examples(clean_examples(all_examples))


def load_source(source: DatasetSource, cache_dir: Path) -> list[Dataset]:
    """Load one dataset source from either local files or the Hugging Face hub."""

    if source.local_path:
        return _load_local_source(source, cache_dir)

    last_error: Exception | None = None
    for candidate in source.candidates:
        try:
            dataset = load_dataset(
                candidate,
                name=source.subset,
                cache_dir=str(cache_dir),
            )
            return _coerce_splits(dataset, source.split_candidates)
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            last_error = exc

    candidate_text = ", ".join(source.candidates)
    raise RuntimeError(
        f"Unable to load dataset '{source.name}'. Tried: {candidate_text}. "
        "Provide a local override or update the dataset id."
    ) from last_error


def _load_local_source(source: DatasetSource, cache_dir: Path) -> list[Dataset]:
    """Load a dataset from a local file or directory."""

    local_path = Path(source.local_path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local dataset path not found: {local_path}")

    if local_path.is_file():
        dataset = load_dataset(
            local_path.suffix.removeprefix("."),
            data_files=str(local_path),
            cache_dir=str(cache_dir),
        )
        return _coerce_splits(dataset, source.split_candidates)

    dataset = load_dataset(str(local_path), cache_dir=str(cache_dir))
    return _coerce_splits(dataset, source.split_candidates)


def _coerce_splits(dataset: Any, split_candidates: Sequence[str]) -> list[Dataset]:
    """Return a list of materialized splits."""

    if isinstance(dataset, Dataset):
        return [dataset]
    if isinstance(dataset, IterableDataset):
        return [Dataset.from_list(list(dataset))]
    if isinstance(dataset, DatasetDict):
        available_splits = [dataset[split] for split in split_candidates if split in dataset]
        if available_splits:
            return available_splits
        return list(dataset.values())
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def _extract_counsel_chat(splits: Sequence[Dataset]) -> list[PreparedExample]:
    """Convert Counsel Chat rows to prompt-response examples."""

    examples: list[PreparedExample] = []
    for split in splits:
        for row in split:
            prompt = _pick_text(row, PROMPT_FIELDS)
            response = _pick_text(row, RESPONSE_FIELDS)
            if not prompt or not response:
                continue
            source_id = _pick_identifier(row, "counsel_chat")
            examples.append(
                PreparedExample(
                    prompt=prompt,
                    response=response,
                    source="counsel_chat",
                    source_id=source_id,
                )
            )
    return examples


def _extract_empathetic_dialogues(splits: Sequence[Dataset]) -> list[PreparedExample]:
    """Convert EmpatheticDialogues to adjacent turn pairs."""

    grouped_turns: dict[str, list[tuple[int, str]]] = {}
    for split in splits:
        for row in split:
            if "utterance" in row:
                conversation_id = str(row.get("conv_id") or row.get("conversation_id") or _pick_identifier(row, "ed"))
                speaker = int(row.get("speaker_idx", 0))
                utterance = _normalize_text(str(row["utterance"]))
                if utterance:
                    grouped_turns.setdefault(conversation_id, []).append((speaker, utterance))
                continue

            conversation = row.get("dialog") or row.get("utterances") or row.get("conversation")
            if isinstance(conversation, Sequence):
                turns = [
                    _normalize_text(str(turn))
                    for turn in conversation
                    if _normalize_text(str(turn))
                ]
                grouped_turns[_pick_identifier(row, "ed")] = list(enumerate(turns))

    examples: list[PreparedExample] = []
    for conversation_id, turns in grouped_turns.items():
        for index in range(len(turns) - 1):
            current_speaker, current_text = turns[index]
            next_speaker, next_text = turns[index + 1]
            if current_speaker == next_speaker:
                continue
            examples.append(
                PreparedExample(
                    prompt=current_text,
                    response=next_text,
                    source="empathetic_dialogues",
                    source_id=f"{conversation_id}:{index}",
                )
            )
    return examples


def _extract_psyqa(splits: Sequence[Dataset]) -> list[PreparedExample]:
    """Convert PsyQA rows to prompt-response examples."""

    examples: list[PreparedExample] = []
    for split in splits:
        for row in split:
            prompt = _build_psyqa_prompt(row)
            response = _build_psyqa_response(row)
            if not prompt or not response:
                continue
            examples.append(
                PreparedExample(
                    prompt=prompt,
                    response=response,
                    source="psyqa",
                    source_id=_pick_identifier(row, "psyqa"),
                )
            )
    return examples


def _build_psyqa_prompt(row: Mapping[str, Any]) -> str:
    """Compose a prompt from the available PsyQA fields."""

    base_prompt = _pick_text(row, PROMPT_FIELDS)
    if not base_prompt:
        return ""

    options = row.get("options")
    if isinstance(options, Mapping):
        option_lines = [f"{key}: {value}" for key, value in options.items()]
        return f"{base_prompt}\n\nOptions:\n" + "\n".join(option_lines)
    if isinstance(options, Sequence) and not isinstance(options, str):
        option_lines = [f"- {item}" for item in options if str(item).strip()]
        if option_lines:
            return f"{base_prompt}\n\nOptions:\n" + "\n".join(option_lines)
    return base_prompt


def _build_psyqa_response(row: Mapping[str, Any]) -> str:
    """Compose a response from answer and rationale fields when present."""

    answer = _pick_text(row, RESPONSE_FIELDS)
    rationale = _pick_text(row, ("explanation", "analysis", "reason"))
    if answer and rationale and rationale not in answer:
        return f"{answer}\n\nWhy: {rationale}"
    return answer


def clean_examples(examples: Iterable[PreparedExample]) -> list[PreparedExample]:
    """Apply text normalization and PII redaction to examples."""

    cleaned: list[PreparedExample] = []
    for example in examples:
        prompt = _clean_text(example.prompt)
        response = _clean_text(example.response)
        if not prompt or not response:
            continue
        cleaned.append(
            PreparedExample(
                prompt=prompt,
                response=response,
                source=example.source,
                source_id=example.source_id,
            )
        )
    return cleaned


def deduplicate_examples(examples: Iterable[PreparedExample]) -> list[PreparedExample]:
    """Drop duplicate prompt-response pairs while preserving order."""

    seen: set[str] = set()
    unique_examples: list[PreparedExample] = []
    for example in examples:
        digest = hashlib.sha256(
            f"{example.prompt}\u241f{example.response}".encode("utf-8")
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique_examples.append(example)
    return unique_examples


def write_jsonl(examples: Sequence[PreparedExample], output_path: Path) -> None:
    """Persist the training corpus to JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_jsonl_record(), ensure_ascii=False) + "\n")


def dataset_summary(examples: Sequence[PreparedExample]) -> dict[str, Any]:
    """Return high-level stats for logging and reporting."""

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.source] = counts.get(example.source, 0) + 1

    return {
        "total_examples": len(examples),
        "by_source": counts,
    }


def _clean_text(text: str) -> str:
    """Normalize whitespace and redact common PII patterns."""

    normalized = _normalize_text(text)
    for pattern, replacement in PII_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized.strip()


def _normalize_text(text: str) -> str:
    """Collapse line endings and repeated whitespace."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pick_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    """Return the first non-empty text field from a row."""

    for field_name in fields:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Sequence) and not isinstance(value, str):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return "\n".join(parts)
    return ""


def _pick_identifier(row: Mapping[str, Any], prefix: str) -> str:
    """Build a stable identifier from the row."""

    for key in ("id", "questionID", "conversation_id", "conv_id", "turn_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raw = json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=False)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _json_safe(value: Any) -> Any:
    """Convert nested dataset values to JSON-safe primitives."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return value
