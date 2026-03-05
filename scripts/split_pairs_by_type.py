#!/usr/bin/env python3
"""
Split sarcasm pairs by type into separate files.

Usage:
    uv run python scripts/split_pairs_by_type.py

Generates:
    - data/processed/sarcasm_pairs_non_to_sarcastic.jsonl
    - data/processed/sarcasm_pairs_sarcastic_to_non_sarcastic.jsonl
"""

import json
from pathlib import Path
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: Path):
    """Save data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} records to: {path}")


def main():
    input_file = Path("data/processed/sarcasm_pairs_step35_clean.jsonl")

    # Output files
    non_to_sarcastic_file = Path("data/processed/sarcasm_pairs_non_to_sarcastic.jsonl")
    sarcastic_to_non_file = Path("data/processed/sarcasm_pairs_sarcastic_to_non.jsonl")

    print(f"Loading pairs from: {input_file}")
    pairs = load_jsonl(input_file)
    print(f"Total pairs loaded: {len(pairs)}")

    # Split by type
    non_to_sarcastic = []
    sarcastic_to_non = []

    for pair in pairs:
        pair_type = pair.get("type", "")
        if pair_type == "non_to_sarcastic":
            non_to_sarcastic.append(pair)
        elif pair_type == "sarcastic_to_non":
            sarcastic_to_non.append(pair)

    print(f"\nSplit results:")
    print(f"  non_to_sarcastic: {len(non_to_sarcastic)}")
    print(f"  sarcastic_to_non: {len(sarcastic_to_non)}")

    # Analyze strategy distribution
    print(f"\nStrategy distribution in non_to_sarcastic:")
    strategies = Counter(p.get("strategy", "unknown") for p in non_to_sarcastic)
    for strategy, count in strategies.most_common():
        print(f"  {strategy}: {count}")

    print(f"\nStrategy distribution in sarcastic_to_non:")
    strategies = Counter(p.get("strategy", "unknown") for p in sarcastic_to_non)
    for strategy, count in strategies.most_common():
        print(f"  {strategy}: {count}")

    # Save to files
    print(f"\nSaving split files...")
    save_jsonl(non_to_sarcastic, non_to_sarcastic_file)
    save_jsonl(sarcastic_to_non, sarcastic_to_non_file)

    print(f"\n✅ Done!")
    print(f"   Total: {len(non_to_sarcastic) + len(sarcastic_to_non)} pairs")


if __name__ == "__main__":
    main()
