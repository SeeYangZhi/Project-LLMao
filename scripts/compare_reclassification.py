#!/usr/bin/env python3
"""
Compare reclassified headlines with original NHDSD labels.
Shows statistics on agreement/disagreement between original and new classifications.

Usage:
    uv run python scripts/compare_reclassification.py
    uv run python scripts/compare_reclassification.py --output-disagreements
    uv run python python scripts/compare_reclassification.py --show-examples 20
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_json(path: str) -> list[dict]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def analyze_agreement(reclassified: list[dict]) -> dict:
    """Analyze agreement between original and reclassified labels."""
    total = len(reclassified)
    agreements = 0
    disagreements = 0

    # Track specific transition patterns
    transitions = Counter()

    # Confidence distribution for disagreements
    disagreement_confidence = Counter()

    disagreement_items = []

    for item in reclassified:
        original = item.get("original_label", 0)
        new = item.get("is_sarcastic", 0)
        confidence = item.get("confidence", "medium")

        if original == new:
            agreements += 1
        else:
            disagreements += 1
            transition_key = f"{original} -> {new}"
            transitions[transition_key] += 1
            disagreement_confidence[confidence] += 1
            disagreement_items.append(item)

    return {
        "total": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": agreements / total * 100 if total > 0 else 0,
        "transitions": transitions,
        "disagreement_confidence": disagreement_confidence,
        "disagreement_items": disagreement_items,
    }


def print_results(stats: dict, show_examples: int = 10):
    """Print analysis results."""
    print("=" * 70)
    print("NHDSD RECLASSIFICATION COMPARISON REPORT")
    print("=" * 70)
    print()

    # Overall statistics
    print("📊 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total headlines:          {stats['total']:,}")
    print(
        f"Agreements:               {stats['agreements']:,} ({stats['agreement_rate']:.2f}%)"
    )
    print(
        f"Disagreements:            {stats['disagreements']:,} ({100 - stats['agreement_rate']:.2f}%)"
    )
    print()

    # Transition patterns
    print("🔄 TRANSITION PATTERNS (Original -> New)")
    print("-" * 70)
    if stats["transitions"]:
        for transition, count in stats["transitions"].most_common():
            pct = (
                count / stats["disagreements"] * 100
                if stats["disagreements"] > 0
                else 0
            )
            label = ""
            if transition == "0 -> 1":
                label = "(Non-sarcastic -> Sarcastic)"
            elif transition == "1 -> 0":
                label = "(Sarcastic -> Non-sarcastic)"
            print(f"  {transition}: {count:>5,} ({pct:>5.1f}%) {label}")
    else:
        print("  No transitions - perfect agreement!")
    print()

    # Confidence on disagreements
    print("🎯 CONFIDENCE LEVELS ON DISAGREEMENTS")
    print("-" * 70)
    if stats["disagreement_confidence"]:
        for conf, count in stats["disagreement_confidence"].most_common():
            pct = (
                count / stats["disagreements"] * 100
                if stats["disagreements"] > 0
                else 0
            )
            print(f"  {conf}: {count:>5,} ({pct:>5.1f}%)")
    print()

    # Example disagreements
    if show_examples > 0 and stats["disagreement_items"]:
        print(
            f"📝 SAMPLE DISAGREEMENTS (showing {min(show_examples, len(stats['disagreement_items']))})"
        )
        print("-" * 70)
        for i, item in enumerate(stats["disagreement_items"][:show_examples], 1):
            orig = item.get("original_label", 0)
            new = item.get("is_sarcastic", 0)
            conf = item.get("confidence", "medium")
            headline = item.get("headline", "")

            orig_label = "Sarcastic" if orig == 1 else "Non-sarcastic"
            new_label = "Sarcastic" if new == 1 else "Non-sarcastic"

            print(f"{i}. {headline}")
            print(
                f"   Original: {orig_label} ({orig}) -> New: {new_label} ({new}) [confidence: {conf}]"
            )
            print()


def save_disagreements(disagreements: list[dict], output_path: str):
    """Save disagreements to JSONL file."""
    with open(output_path, "w") as f:
        for item in disagreements:
            f.write(json.dumps(item) + "\n")
    print(f"💾 Saved {len(disagreements)} disagreements to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reclassified headlines with original NHDSD labels"
    )
    parser.add_argument(
        "--reclassified",
        type=str,
        default="data/processed/nhdsd_reclassified.jsonl",
        help="Path to reclassified JSONL file",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=10,
        help="Number of example disagreements to show (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--output-disagreements",
        type=str,
        metavar="PATH",
        nargs="?",
        const="data/processed/label_disagreements.jsonl",
        help="Save disagreements to file (default: data/processed/label_disagreements.jsonl)",
    )
    args = parser.parse_args()

    # Load reclassified data
    print(f"Loading reclassified data from: {args.reclassified}")
    reclassified = load_jsonl(args.reclassified)

    # Analyze
    stats = analyze_agreement(reclassified)

    # Print results
    print_results(stats, show_examples=args.show_examples)

    # Save disagreements if requested
    if args.output_disagreements and stats["disagreement_items"]:
        save_disagreements(stats["disagreement_items"], args.output_disagreements)

    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
