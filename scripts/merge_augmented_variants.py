#!/usr/bin/env python3
"""
Merge augmented strategy variants into non_to_sarcastic pairs file.

Usage:
    uv run python scripts/merge_augmented_variants.py

Takes:
    - data/processed/sarcasm_pairs_non_to_sarcastic.jsonl (original pairs)
    - data/processed/sarcasm_pairs_strategy_augmented.jsonl (5 variants per source)

Produces:
    - data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl
      (original + 5 augmented variants = 6 variants per source)
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


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
    # Input files
    original_file = Path("data/processed/sarcasm_pairs_non_to_sarcastic.jsonl")
    augmented_file = Path("data/processed/sarcasm_pairs_strategy_augmented.jsonl")

    # Output file
    output_file = Path("data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl")

    print(f"Loading original pairs from: {original_file}")
    original_pairs = load_jsonl(original_file)
    print(f"  Loaded: {len(original_pairs)} original pairs")

    print(f"\nLoading augmented variants from: {augmented_file}")
    augmented = load_jsonl(augmented_file)
    print(f"  Loaded: {len(augmented)} augmented variants")

    # Group augmented by source headline
    by_source = defaultdict(list)
    for item in augmented:
        key = item.get("original_headline", "")
        by_source[key].append(item)

    print(f"  Unique sources in augmented: {len(by_source)}")

    # Convert original pairs to standard format and merge with augmented
    complete_pairs = []
    missing_augmented = []

    for orig in original_pairs:
        headline = orig.get("original_headline", "")
        existing_strategy = orig.get("strategy", "")

        # Add the original as the first variant (with its existing strategy)
        complete_pairs.append(
            {
                "original_headline": headline,
                "non_sarcastic_source": headline,
                "generated_headline": orig.get("generated_headline", ""),
                "strategy": existing_strategy,
                "existing_strategy": existing_strategy,
                "type": "non_to_sarcastic",
                "variant_type": "original",
                "model_used": orig.get("model_used", ""),
                "article_link": orig.get("article_link", ""),
            }
        )

        # Add augmented variants
        variants = by_source.get(headline, [])
        for variant in variants:
            complete_pairs.append(
                {
                    "original_headline": headline,
                    "non_sarcastic_source": headline,
                    "generated_headline": variant.get("generated_headline", ""),
                    "strategy": variant.get("strategy", ""),
                    "existing_strategy": existing_strategy,
                    "type": "non_to_sarcastic",
                    "variant_type": "augmented",
                    "model_used": variant.get("model_used", ""),
                    "article_link": variant.get("article_link", ""),
                }
            )

        if not variants:
            missing_augmented.append(headline)

    print(f"\nMerge results:")
    print(f"  Total complete pairs: {len(complete_pairs)}")
    print(f"  Sources missing augmented variants: {len(missing_augmented)}")

    if missing_augmented:
        print(f"  (First 5 missing: {missing_augmented[:5]})")

    # Verify strategy distribution
    print(f"\nFinal strategy distribution:")
    strategies = Counter(p.get("strategy", "unknown") for p in complete_pairs)
    for strategy, count in strategies.most_common():
        print(f"  {strategy}: {count}")

    # Verify variant counts per source
    sources_with_counts = defaultdict(int)
    for p in complete_pairs:
        sources_with_counts[p.get("original_headline", "")] += 1

    complete_6 = sum(1 for count in sources_with_counts.values() if count == 6)
    incomplete = sum(1 for count in sources_with_counts.values() if count != 6)

    print(f"\nPer-source variant counts:")
    print(f"  Sources with 6 variants: {complete_6}")
    print(f"  Sources with != 6 variants: {incomplete}")

    if incomplete != 0:
        print(f"\n❌ ERROR: {incomplete} sources do not have exactly 6 variants!")
        print(f"   Complete (6/6): {complete_6}")
        print(f"   Incomplete:     {incomplete}")
        print(f"   Expected total: {len(original_pairs) * 6} records ({len(original_pairs)} sources × 6)")
        print(f"   Actual total:   {len(complete_pairs)} records")
        print(f"   Output NOT saved. Fix incomplete sources before merging.")
        raise SystemExit(1)

    # Save merged file
    print(f"\nSaving complete dataset...")
    save_jsonl(complete_pairs, output_file)

    print(f"\n✅ Done!")
    print(f"   Original pairs: {len(original_pairs)}")
    print(f"   Augmented variants: {len(augmented)}")
    print(f"   Total in complete file: {len(complete_pairs)}")
    print(f"   Expected (if all complete): {len(original_pairs) * 6}")


if __name__ == "__main__":
    main()
