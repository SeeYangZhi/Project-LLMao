#!/usr/bin/env python3
"""Analyze cross-validation results: compare original, stepfun, and secondary model labels.

Usage:
    uv run python scripts/analyze_cross_validation.py
    uv run python scripts/analyze_cross_validation.py --output-csv
"""

import argparse
import json


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_votes(data: list[dict]) -> dict:
    """Analyze three-way voting patterns."""
    total = len(data)

    # Vote outcomes
    original_and_secondary_agree = 0
    stepfun_and_secondary_agree = 0

    # By direction (originally sarcastic vs non-sarcastic)
    originally_sarcastic = {
        "total": 0,
        "original_and_secondary": 0,
        "stepfun_and_secondary": 0,
    }
    originally_non_sarcastic = {
        "total": 0,
        "original_and_secondary": 0,
        "stepfun_and_secondary": 0,
    }

    # Confidence analysis
    high_confidence_both = 0
    mixed_confidence = 0

    # Source tracking
    onion_disagreements = []
    huffpost_disagreements = []

    for item in data:
        original = item.get("original_label", 0)
        stepfun = item.get("stepfun_label", 0)
        secondary = item.get("is_sarcastic", 0)
        stepfun_conf = item.get("stepfun_confidence", "medium")
        secondary_conf = item.get("confidence", "medium")

        # Track by original source
        article_link = item.get("article_link", "")
        if "theonion.com" in article_link:
            onion_disagreements.append(item)
        elif "huffingtonpost.com" in article_link:
            huffpost_disagreements.append(item)

        # Determine vote outcome
        if original == secondary:
            original_and_secondary_agree += 1
            if original == 1:
                originally_sarcastic["original_and_secondary"] += 1
                originally_sarcastic["total"] += 1
            else:
                originally_non_sarcastic["original_and_secondary"] += 1
                originally_non_sarcastic["total"] += 1
        elif stepfun == secondary:
            stepfun_and_secondary_agree += 1
            if original == 1:
                originally_sarcastic["stepfun_and_secondary"] += 1
                originally_sarcastic["total"] += 1
            else:
                originally_non_sarcastic["stepfun_and_secondary"] += 1
                originally_non_sarcastic["total"] += 1

        # Confidence tracking
        if stepfun_conf == "high" and secondary_conf == "high":
            high_confidence_both += 1
        else:
            mixed_confidence += 1

    return {
        "total": total,
        "original_and_secondary_agree": original_and_secondary_agree,
        "stepfun_and_secondary_agree": stepfun_and_secondary_agree,
        "originally_sarcastic": originally_sarcastic,
        "originally_non_sarcastic": originally_non_sarcastic,
        "high_confidence_both": high_confidence_both,
        "mixed_confidence": mixed_confidence,
        "onion_count": len(onion_disagreements),
        "huffpost_count": len(huffpost_disagreements),
    }


def generate_report(stats: dict) -> dict:
    """Generate structured comparison report."""
    total = stats["total"]

    # Calculate percentages
    original_secondary_pct = (
        stats["original_and_secondary_agree"] / total * 100 if total > 0 else 0
    )
    stepfun_secondary_pct = (
        stats["stepfun_and_secondary_agree"] / total * 100 if total > 0 else 0
    )

    # By direction
    sarc_total = stats["originally_sarcastic"]["total"]
    non_sarc_total = stats["originally_non_sarcastic"]["total"]

    # Estimate NHDSD error rate
    # If Secondary model agrees with StepFun, and they both disagree with original, original is wrong
    estimated_mislabels = stats["stepfun_and_secondary_agree"]
    estimated_error_rate = estimated_mislabels / total * 100 if total > 0 else 0

    return {
        "metadata": {
            "total_disagreements": total,
            "cross_validated": total,
            "models": [
                "original_nhdsd",
                "stepfun/step-3.5-flash:free",
                "nvidia/nemotron-3-nano-30b-a3b:free",
            ],
        },
        "vote_outcomes": {
            "original_and_secondary_agree": {
                "count": stats["original_and_secondary_agree"],
                "pct": round(original_secondary_pct, 2),
                "interpretation": "StepFun likely wrong; original label probably correct",
            },
            "stepfun_and_secondary_agree": {
                "count": stats["stepfun_and_secondary_agree"],
                "pct": round(stepfun_secondary_pct, 2),
                "interpretation": "Original label likely wrong; genuine mislabel in NHDSD",
            },
        },
        "by_original_label": {
            "originally_sarcastic": {
                "total": sarc_total,
                "secondary_agrees_with_original": stats["originally_sarcastic"][
                    "original_and_secondary"
                ],
                "secondary_agrees_with_stepfun": stats["originally_sarcastic"][
                    "stepfun_and_secondary"
                ],
            },
            "originally_non_sarcastic": {
                "total": non_sarc_total,
                "secondary_agrees_with_original": stats["originally_non_sarcastic"][
                    "original_and_secondary"
                ],
                "secondary_agrees_with_stepfun": stats["originally_non_sarcastic"][
                    "stepfun_and_secondary"
                ],
            },
        },
        "by_source": {
            "theonion_disagreements": stats["onion_count"],
            "huffpost_disagreements": stats["huffpost_count"],
        },
        "confidence_analysis": {
            "high_confidence_both": stats["high_confidence_both"],
            "mixed_confidence": stats["mixed_confidence"],
        },
        "ground_truth_assessment": {
            "estimated_nhdsd_error_rate": round(estimated_error_rate, 2),
            "estimated_mislabels": estimated_mislabels,
            "interpretation": f"~{estimated_error_rate:.1f}% of disagreements are likely true NHDSD errors",
        },
    }


def print_report(report: dict):
    """Print formatted report to console."""
    print("=" * 70)
    print("CROSS-VALIDATION ANALYSIS REPORT")
    print("=" * 70)
    print()

    # Overview
    print("📊 OVERVIEW")
    print("-" * 70)
    print(
        f"Total disagreements analyzed: {report['metadata']['total_disagreements']:,}"
    )
    print()

    # Vote outcomes
    print("🗳️  VOTE OUTCOMES")
    print("-" * 70)
    orig_secondary = report["vote_outcomes"]["original_and_secondary_agree"]
    stepfun_secondary = report["vote_outcomes"]["stepfun_and_secondary_agree"]

    print(
        f"Secondary model agrees with Original:  {orig_secondary['count']:>5,} ({orig_secondary['pct']:>5.1f}%)"
    )
    print(f"  → {orig_secondary['interpretation']}")
    print()
    print(
        f"Secondary model agrees with StepFun:   {stepfun_secondary['count']:>5,} ({stepfun_secondary['pct']:>5.1f}%)"
    )
    print(f"  → {stepfun_secondary['interpretation']}")
    print()

    # By original label
    print("📈 BY ORIGINAL LABEL")
    print("-" * 70)
    sarc = report["by_original_label"]["originally_sarcastic"]
    non_sarc = report["by_original_label"]["originally_non_sarcastic"]

    print(f"Originally Sarcastic:      {sarc['total']:>5,}")
    print(
        f"  Secondary model agrees w/ Original: {sarc['secondary_agrees_with_original']:>5,}"
    )
    print(
        f"  Secondary model agrees w/ StepFun:  {sarc['secondary_agrees_with_stepfun']:>5,}"
    )
    print()
    print(f"Originally Non-Sarcastic:  {non_sarc['total']:>5,}")
    print(
        f"  Secondary model agrees w/ Original: {non_sarc['secondary_agrees_with_original']:>5,}"
    )
    print(
        f"  Secondary model agrees w/ StepFun:  {non_sarc['secondary_agrees_with_stepfun']:>5,}"
    )
    print()

    # By source
    print("📰 BY SOURCE")
    print("-" * 70)
    print(
        f"TheOnion disagreements:    {report['by_source']['theonion_disagreements']:>5,}"
    )
    print(
        f"HuffPost disagreements:    {report['by_source']['huffpost_disagreements']:>5,}"
    )
    print()

    # Ground truth assessment
    print("🎯 GROUND TRUTH ASSESSMENT")
    print("-" * 70)
    assessment = report["ground_truth_assessment"]
    print(
        f"Estimated NHDSD errors:    {assessment['estimated_mislabels']:>5,} ({assessment['estimated_nhdsd_error_rate']:.1f}%)"
    )
    print(f"  → {assessment['interpretation']}")
    print()

    # Confidence
    print("💪 CONFIDENCE ANALYSIS")
    print("-" * 70)
    conf = report["confidence_analysis"]
    print(f"Both models high confidence: {conf['high_confidence_both']:>5,}")
    print(f"Mixed/lower confidence:      {conf['mixed_confidence']:>5,}")
    print()

    print("=" * 70)


def save_csv(data: list[dict], output_path: str):
    """Save detailed results to CSV for manual review."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "headline",
                "original_label",
                "stepfun_label",
                "secondary_label",
                "stepfun_confidence",
                "secondary_confidence",
                "vote_outcome",
                "article_link",
            ]
        )

        for item in data:
            original = item.get("original_label", 0)
            stepfun = item.get("stepfun_label", 0)
            secondary = item.get("is_sarcastic", 0)

            # Determine outcome
            if original == secondary:
                outcome = "secondary_agrees_original"
            elif stepfun == secondary:
                outcome = "secondary_agrees_stepfun"
            else:
                outcome = "unknown"

            writer.writerow(
                [
                    item.get("headline", ""),
                    original,
                    stepfun,
                    secondary,
                    item.get("stepfun_confidence", "medium"),
                    item.get("confidence", "medium"),
                    outcome,
                    item.get("article_link", ""),
                ]
            )

    print(f"💾 Saved CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-validation results")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/cross_validation_secondary.jsonl",
        help="Path to cross-validation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/cross_validation_comparison.json",
        help="Path for JSON output",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        metavar="PATH",
        nargs="?",
        const="data/processed/cross_validation_detailed.csv",
        help="Also save detailed CSV for manual review",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading cross-validation data from: {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded: {len(data)} records")

    # Analyze
    stats = analyze_votes(data)
    report = generate_report(stats)

    # Print report
    print_report(report)

    # Save JSON report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Saved JSON report to: {args.output}")

    # Save CSV if requested
    if args.output_csv:
        save_csv(data, args.output_csv)


if __name__ == "__main__":
    main()
