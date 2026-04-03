"""Classify scraped Onion headlines using our trained sarcasm classifier.

Loads headlines from data/raw/onion_headlines.jsonl, runs them through
loyongzhe/sarcasm-classifier-binary, and reports statistics.

Usage:
    uv run python scripts/data_prep/classify_onion_headlines.py
    uv run python scripts/data_prep/classify_onion_headlines.py --output data/processed/onion_headlines_classified.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "onion_headlines.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "onion_headlines_classified.jsonl"
MODEL_NAME = "loyongzhe/sarcasm-classifier-binary"
BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser(description="Classify Onion headlines for sarcasm")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Load headlines
    headlines = []
    with open(args.input) as f:
        for line in f:
            headlines.append(json.loads(line))
    print(f"Loaded {len(headlines)} headlines from {args.input}")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Detect label mapping
    id2label = model.config.id2label
    print(f"Labels: {id2label}")

    sarcastic_idx = None
    for idx, label in id2label.items():
        label_lower = label.lower().replace(" ", "_")
        if label_lower in ("sarcastic", "sarc", "1") or (
            "sarcastic" in label_lower and "non" not in label_lower
        ):
            sarcastic_idx = int(idx)
            break
    if sarcastic_idx is None:
        sarcastic_idx = 1
    print(f"Sarcastic label index: {sarcastic_idx}")

    # Classify in batches
    results = []
    texts = [h["headline"] for h in headlines]

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        for j, (headline_rec, prob) in enumerate(zip(headlines[i : i + len(batch)], probs)):
            sarc_prob = prob[sarcastic_idx].item()
            pred_idx = prob.argmax().item()
            results.append({
                **headline_rec,
                "predicted_label": id2label[pred_idx],
                "sarcasm_probability": round(sarc_prob, 4),
            })

        done = min(i + args.batch_size, len(texts))
        print(f"  Classified {done}/{len(texts)}", flush=True)

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics
    sarc_count = sum(1 for r in results if r["sarcasm_probability"] > 0.5)
    non_sarc_count = len(results) - sarc_count
    avg_prob = sum(r["sarcasm_probability"] for r in results) / len(results)
    sarc_probs = sorted([r["sarcasm_probability"] for r in results], reverse=True)

    print(f"\n{'='*50}")
    print(f"Results: {len(results)} headlines classified")
    print(f"  Sarcastic:     {sarc_count} ({100*sarc_count/len(results):.1f}%)")
    print(f"  Non-sarcastic: {non_sarc_count} ({100*non_sarc_count/len(results):.1f}%)")
    print(f"  Avg sarcasm prob: {avg_prob:.4f}")
    print(f"  Median sarcasm prob: {sarc_probs[len(sarc_probs)//2]:.4f}")
    print(f"\nTop 10 most sarcastic:")
    for r in sorted(results, key=lambda x: x["sarcasm_probability"], reverse=True)[:10]:
        print(f"  [{r['sarcasm_probability']:.3f}] {r['headline']}")
    print(f"\nTop 10 least sarcastic:")
    for r in sorted(results, key=lambda x: x["sarcasm_probability"])[:10]:
        print(f"  [{r['sarcasm_probability']:.3f}] {r['headline']}")
    print(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main()
