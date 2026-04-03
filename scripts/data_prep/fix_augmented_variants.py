#!/usr/bin/env python3
"""
Fix augmented strategy variants to ensure exactly 5 unique strategies per source.

Usage:
    uv run python scripts/fix_augmented_variants.py
"""

import json
from collections import defaultdict
from pathlib import Path


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
    augmented_file = Path("data/processed/sarcasm_pairs_strategy_augmented.jsonl")
    output_file = Path("data/processed/sarcasm_pairs_strategy_augmented_fixed.jsonl")

    print(f"Loading augmented variants from: {augmented_file}")
    augmented = load_jsonl(augmented_file)
    print(f"  Loaded: {len(augmented)} records")

    # Group by source
    by_source = defaultdict(list)
    for item in augmented:
        key = item.get("original_headline", "")
        by_source[key].append(item)

    print(f"  Unique sources: {len(by_source)}")

    # Fix: deduplicate by strategy, keep only 5 per source
    fixed = []
    all_strategies = [
        "sarcasm",
        "irony",
        "satire",
        "understatement",
        "overstatement",
        "rhetorical_question",
    ]

    issues_found = 0
    for headline, variants in by_source.items():
        # Get existing strategy from first variant
        existing_strategy = variants[0].get("existing_strategy", "") if variants else ""

        # Deduplicate by strategy, keep first occurrence
        seen_strategies = set()
        unique_variants = []

        for v in variants:
            strategy = v.get("strategy", "")
            # Skip if we've seen this strategy already
            if strategy in seen_strategies:
                continue
            # Skip if this is the existing strategy (shouldn't be in augmented)
            if strategy == existing_strategy:
                continue
            seen_strategies.add(strategy)
            unique_variants.append(v)

        # Check if we have the right number
        if len(unique_variants) != 5:
            issues_found += 1
            missing = set(all_strategies) - {existing_strategy} - seen_strategies
            extra = len(unique_variants) - 5
            if missing:
                print(
                    f"  Warning: '{headline[:50]}...' missing {len(missing)} strategies: {missing}"
                )
            if extra > 0:
                print(
                    f"  Warning: '{headline[:50]}...' has {extra} extra variants, truncating to 5"
                )
                unique_variants = unique_variants[:5]

        fixed.extend(unique_variants)

    print(f"\nFix results:")
    print(f"  Original records: {len(augmented)}")
    print(f"  Fixed records: {len(fixed)}")
    print(f"  Sources with issues: {issues_found}")

    # Verify
    by_source_fixed = defaultdict(list)
    for item in fixed:
        key = item.get("original_headline", "")
        by_source_fixed[key].append(item)

    perfect = sum(1 for v in by_source_fixed.values() if len(v) == 5)
    print(f"  Sources with exactly 5 variants: {perfect}/{len(by_source_fixed)}")

    # Save fixed file
    print(f"\nSaving fixed file...")
    save_jsonl(fixed, output_file)

    print("\n✅ Done!")
    print(f"   Replace original with: mv {output_file} {augmented_file}")


if __name__ == "__main__":
    main()
