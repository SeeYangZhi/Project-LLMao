#!/usr/bin/env python3
"""
Create stratified train/val/test splits from the complete dataset.

Splits at the SOURCE level (all 6 strategy variants of a source stay together)
to prevent data leakage. Stratifies by the original strategy of each source.

Usage:
    uv run python scripts/create_train_val_test_splits.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Config
INPUT_FILE = "data/processed/sarcasm_pairs_non_to_sarcastic_complete.jsonl"
OUTPUT_DIR = Path("data/splits")
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


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
    random.seed(SEED)

    print(f"Loading complete dataset from {INPUT_FILE}...")
    all_records = load_jsonl(INPUT_FILE)
    print(f"  Total records: {len(all_records)}")

    # Group records by source headline
    by_source = defaultdict(list)
    for record in all_records:
        source = record.get("original_headline", "")
        by_source[source].append(record)

    total_sources = len(by_source)
    print(f"  Unique sources: {total_sources}")

    # Verify all sources have 6 variants
    variant_counts = Counter(len(v) for v in by_source.values())
    print(f"  Variant count distribution: {dict(variant_counts)}")

    if variant_counts.get(6, 0) != total_sources:
        print("  WARNING: Not all sources have exactly 6 variants!")

    # Get the original strategy for each source (for stratification)
    # Use the "original" variant_type or the existing_strategy field
    source_strategies = {}
    for source, records in by_source.items():
        # Find the original (non-augmented) record
        original_records = [r for r in records if r.get("variant_type") == "original"]
        if original_records:
            source_strategies[source] = original_records[0].get("strategy", "unknown")
        else:
            # Fallback: use existing_strategy from any record
            source_strategies[source] = records[0].get("existing_strategy", "unknown")

    # Group sources by their original strategy for stratified splitting
    strategy_sources = defaultdict(list)
    for source, strategy in source_strategies.items():
        strategy_sources[strategy].append(source)

    print(f"\n  Sources by original strategy:")
    for strategy, sources in sorted(strategy_sources.items()):
        print(f"    {strategy}: {len(sources)}")

    # Stratified split: split each strategy group independently
    train_sources = []
    val_sources = []
    test_sources = []

    for strategy, sources in strategy_sources.items():
        random.shuffle(sources)
        n = len(sources)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Test gets the remainder to avoid rounding losses
        train_sources.extend(sources[:n_train])
        val_sources.extend(sources[n_train : n_train + n_val])
        test_sources.extend(sources[n_train + n_val :])

    train_set = set(train_sources)
    val_set = set(val_sources)
    test_set = set(test_sources)

    # Verify no overlap
    assert len(train_set & val_set) == 0, "Train/val overlap!"
    assert len(train_set & test_set) == 0, "Train/test overlap!"
    assert len(val_set & test_set) == 0, "Val/test overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == total_sources, (
        "Source count mismatch!"
    )

    # Collect records for each split
    train_records = []
    val_records = []
    test_records = []

    for source, records in by_source.items():
        if source in train_set:
            train_records.extend(records)
        elif source in val_set:
            val_records.extend(records)
        elif source in test_set:
            test_records.extend(records)

    # Shuffle within each split
    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    print(f"\nSplit results:")
    print(
        f"  Train: {len(train_sources)} sources, {len(train_records)} records ({100 * len(train_sources) / total_sources:.1f}%)"
    )
    print(
        f"  Val:   {len(val_sources)} sources, {len(val_records)} records ({100 * len(val_sources) / total_sources:.1f}%)"
    )
    print(
        f"  Test:  {len(test_sources)} sources, {len(test_records)} records ({100 * len(test_sources) / total_sources:.1f}%)"
    )

    # Verify strategy distribution in each split
    for split_name, split_records in [
        ("Train", train_records),
        ("Val", val_records),
        ("Test", test_records),
    ]:
        strategy_dist = Counter(r.get("strategy", "unknown") for r in split_records)
        print(f"\n  {split_name} strategy distribution:")
        for strategy, count in strategy_dist.most_common():
            print(f"    {strategy}: {count}")

    # Save splits
    print(f"\nSaving splits to {OUTPUT_DIR}/...")
    save_jsonl(train_records, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_records, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_records, OUTPUT_DIR / "test.jsonl")

    # Save split metadata
    metadata = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "source_file": INPUT_FILE,
        "total_sources": total_sources,
        "total_records": len(all_records),
        "splits": {
            "train": {
                "sources": len(train_sources),
                "records": len(train_records),
            },
            "val": {
                "sources": len(val_sources),
                "records": len(val_records),
            },
            "test": {
                "sources": len(test_sources),
                "records": len(test_records),
            },
        },
        "strategy_distribution": {
            strategy: len(sources)
            for strategy, sources in sorted(strategy_sources.items())
        },
    }

    metadata_path = OUTPUT_DIR / "split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to: {metadata_path}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
