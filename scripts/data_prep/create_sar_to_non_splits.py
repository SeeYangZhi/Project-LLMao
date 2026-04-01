"""Create train/val/test splits for sarcastic-to-non-sarcastic data.

Splits data/processed/sarcasm_pairs_sarcastic_to_non.jsonl (13,588 records)
into 80/10/10 train/val/test with stratification by strategy.

Usage:
    uv run python scripts/data_prep/create_sar_to_non_splits.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sarcasm_pairs_sarcastic_to_non.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits" / "sar_to_non"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def stratified_split(records: list[dict], ratios: dict[str, float], seed: int) -> dict[str, list[dict]]:
    """Split records into train/val/test, stratified by strategy."""
    rng = random.Random(seed)

    by_strategy: dict[str, list[dict]] = {}
    for r in records:
        by_strategy.setdefault(r["strategy"], []).append(r)

    splits: dict[str, list[dict]] = {k: [] for k in ratios}

    for strategy, items in sorted(by_strategy.items()):
        rng.shuffle(items)
        n = len(items)
        train_end = int(n * ratios["train"])
        val_end = train_end + int(n * ratios["val"])

        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])

    for items in splits.values():
        rng.shuffle(items)

    return splits


def main():
    records = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(records)} records from {INPUT_FILE.name}")

    splits = stratified_split(records, SPLIT_RATIOS, SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "total_records": len(records),
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "splits": {},
        "strategy_distribution": {},
    }

    for split_name, items in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        strategy_counts = Counter(r["strategy"] for r in items)
        metadata["splits"][split_name] = {
            "count": len(items),
            "strategy_distribution": dict(sorted(strategy_counts.items())),
        }
        print(f"  {split_name}: {len(items)} records → {out_path.name}")

    overall_dist = Counter(r["strategy"] for r in records)
    metadata["strategy_distribution"] = dict(sorted(overall_dist.items()))

    meta_path = OUTPUT_DIR / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata written to {meta_path.name}")
    print(f"Strategy distribution: {dict(sorted(overall_dist.items()))}")


if __name__ == "__main__":
    main()
