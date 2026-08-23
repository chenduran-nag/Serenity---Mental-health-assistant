"""Build the Serenity training corpus from open mental health datasets."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import DataPrepConfig, dataset_summary, build_training_corpus, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for dataset preparation."""

    parser = argparse.ArgumentParser(
        description="Download, clean, and merge Serenity training datasets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "mental_health_dataset.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "cache",
        help="Dataset cache directory.",
    )
    parser.add_argument(
        "--counsel-chat-local",
        type=str,
        default=None,
        help="Optional local file or directory for Counsel Chat.",
    )
    parser.add_argument(
        "--empathetic-dialogues-local",
        type=str,
        default=None,
        help="Optional local file or directory for EmpatheticDialogues.",
    )
    parser.add_argument(
        "--psyqa-local",
        type=str,
        default=None,
        help="Optional local file or directory for PsyQA.",
    )
    parser.add_argument(
        "--max-examples-per-source",
        type=int,
        default=None,
        help=(
            "Cap each source at N examples, sampled deterministically. "
            "Without it EmpatheticDialogues outnumbers Counsel Chat about 29:1."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "data" / "processed" / "dataset_summary.json",
        help="Where to write corpus summary metadata.",
    )
    return parser


def main() -> int:
    """Run the data preparation workflow."""

    parser = build_parser()
    args = parser.parse_args()

    config = DataPrepConfig(
        output_path=args.output,
        cache_dir=args.cache_dir,
        max_examples_per_source=args.max_examples_per_source,
    )
    config.counsel_chat.local_path = args.counsel_chat_local
    config.empathetic_dialogues.local_path = args.empathetic_dialogues_local
    config.psyqa.local_path = args.psyqa_local

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = build_training_corpus(config)
    write_jsonl(result.examples, args.output)

    summary = dataset_summary(result.examples)
    summary["sources"] = [asdict(outcome) for outcome in result.outcomes]
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for outcome in result.skipped:
        print(f"WARNING: skipped {outcome.name}: {outcome.error}", file=sys.stderr)

    print(json.dumps({"output": str(args.output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
