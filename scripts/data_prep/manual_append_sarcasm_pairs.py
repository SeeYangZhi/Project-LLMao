#!/usr/bin/env python3
"""
Manually append sarcasm style-transfer pairs from remaining headlines.

Usage:
    uv run python scripts/manual_append_sarcasm_pairs.py
    uv run python scripts/manual_append_sarcasm_pairs.py --output data/processed/sarcasm_pairs_step35_clean.jsonl

Note: there was a bug in the filename so its appended manually like so:
    `cat data/processed/sarcasm_pairs_step_35_clean.jsonl >> data/processed/sarcasm_pairs_step35_clean.jsonl`
"""

import argparse
import json
from pathlib import Path

VALID_STRATEGIES = {
    "sarcasm",
    "irony",
    "satire",
    "understatement",
    "overstatement",
    "rhetorical_question",
}

MODEL_NAME = "human/nus-student-camille"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_pair_type(is_sarcastic: int) -> str:
    return "sarcastic_to_non" if is_sarcastic == 1 else "non_to_sarcastic"


def get_target_label(is_sarcastic: int) -> str:
    return "non-sarcastic" if is_sarcastic == 1 else "sarcastic"


def prompt_strategy() -> str:
    while True:
        strategy = input(
            "Strategy [sarcasm|irony|satire|understatement|overstatement|rhetorical_question]: "
        ).strip()
        if strategy in VALID_STRATEGIES:
            return strategy
        print("Invalid strategy. Please enter one of the listed values.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually add sarcasm transfer pairs from remaining_headlines.jsonl"
    )
    parser.add_argument(
        "--remaining",
        default="data/processed/remaining_headlines.jsonl",
        help="Path to remaining headlines JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/processed/sarcasm_pairs_step_35_clean.jsonl",
        help="Path to output sarcasm pairs JSONL",
    )
    args = parser.parse_args()

    remaining_path = Path(args.remaining)
    output_path = Path(args.output)

    if not remaining_path.exists():
        raise FileNotFoundError(f"Remaining headlines file not found: {remaining_path}")

    items = load_jsonl(remaining_path)
    if not items:
        print("No remaining headlines found.")
        return

    print(f"Loaded {len(items)} remaining headlines from {remaining_path}")
    print(f"Appending to: {output_path}")
    print("Enter headline index to annotate, or 'q' to quit.")

    while True:
        raw_index = input("\nIndex> ").strip().lower()
        if raw_index in {"q", "quit", "exit"}:
            print("Done.")
            break

        if not raw_index.isdigit():
            print("Please enter a numeric index or 'q'.")
            continue

        index = int(raw_index)
        if index < 0 or index >= len(items):
            print(f"Index out of range. Valid range: 0 to {len(items) - 1}")
            continue

        item = items[index]
        source_headline = item.get("headline", "").strip()
        is_sarcastic = int(item.get("is_sarcastic", 0))
        pair_type = get_pair_type(is_sarcastic)
        target_label = get_target_label(is_sarcastic)

        print(f"\n[{index}] is_sarcastic={is_sarcastic}")
        print(f"Source headline: {source_headline}")
        generated = input(f"Enter {target_label} headline: ").strip()
        if not generated:
            print("Skipped (empty headline).")
            continue

        strategy = prompt_strategy()

        record = {
            "original_headline": source_headline,
            "generated_headline": generated,
            "strategy": strategy,
            "type": pair_type,
            "model_used": MODEL_NAME,
            "article_link": item.get("article_link", ""),
        }

        append_jsonl(output_path, record)
        print("Appended 1 record.")


if __name__ == "__main__":
    main()
