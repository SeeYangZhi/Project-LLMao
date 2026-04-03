"""Evaluate BART sar-to-non model on sarcastic Onion headlines.

Loads sarcastic headlines from onion_headlines_classified.jsonl,
generates non-sarcastic versions using the BART checkpoint, then
classifies the outputs to check if they are actually non-sarcastic.

Usage:
    uv run python scripts/eval_bart_onion.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADLINES_PATH = PROJECT_ROOT / "data" / "processed" / "onion_headlines_classified.jsonl"
DEFAULT_BART_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bart-base" / "sar-to-non" / "final"
CLASSIFIER_MODEL = "loyongzhe/sarcasm-classifier-binary"
BATCH_SIZE = 32


def load_sarcastic_headlines(path: Path) -> list[dict]:
    headlines = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["sarcasm_probability"] > 0.5:
                headlines.append(rec)
    return headlines


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_BART_CHECKPOINT),
                        help="BART model path or HF name (default: fine-tuned checkpoint)")
    args = parser.parse_args()

    BART_CHECKPOINT = args.model

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load BART model
    print(f"Loading BART from {BART_CHECKPOINT}...")
    bart_tokenizer = BartTokenizer.from_pretrained(BART_CHECKPOINT)
    bart_model = BartForConditionalGeneration.from_pretrained(BART_CHECKPOINT).to(device)
    bart_model.eval()

    # Load classifier
    print(f"Loading classifier {CLASSIFIER_MODEL}...")
    cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    cls_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL).to(device)
    cls_model.eval()

    # Detect sarcastic label index
    id2label = cls_model.config.id2label
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
    print(f"Classifier labels: {id2label}, sarcastic_idx={sarcastic_idx}")

    # Load sarcastic headlines
    headlines = load_sarcastic_headlines(HEADLINES_PATH)
    print(f"Loaded {len(headlines)} sarcastic Onion headlines\n")

    # Generate and classify
    results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i : i + BATCH_SIZE]
        texts = [h["headline"] for h in batch]

        # BART generation (no prefix for BART)
        inputs = bart_tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen_ids = bart_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                length_penalty=1.0,
            )
        generated = bart_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Classify generated outputs
        cls_inputs = cls_tokenizer(
            generated, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = cls_model(**cls_inputs).logits
            probs = torch.softmax(logits, dim=-1)

        for h, gen_text, prob in zip(batch, generated, probs):
            sarc_prob = prob[sarcastic_idx].item()
            results.append({
                "input": h["headline"],
                "input_sarc_prob": h["sarcasm_probability"],
                "output": gen_text,
                "output_sarc_prob": round(sarc_prob, 4),
                "is_non_sarcastic": sarc_prob <= 0.5,
            })

        done = min(i + BATCH_SIZE, len(headlines))
        print(f"  Processed {done}/{len(headlines)}", flush=True)

    # Statistics
    total = len(results)
    success = sum(1 for r in results if r["is_non_sarcastic"])
    avg_output_sarc = sum(r["output_sarc_prob"] for r in results) / total
    identical = sum(1 for r in results if r["input"].strip().lower() == r["output"].strip().lower())

    print(f"\n{'='*60}")
    print(f"Total sarcastic inputs:        {total}")
    print(f"Successfully de-sarcasmed:     {success} ({100*success/total:.1f}%)")
    print(f"Still sarcastic:               {total - success} ({100*(total-success)/total:.1f}%)")
    print(f"Avg output sarcasm prob:       {avg_output_sarc:.4f}")
    print(f"Identical to input:            {identical} ({100*identical/total:.1f}%)")

    print(f"\n--- Sample outputs (first 20) ---")
    for r in results[:20]:
        status = "OK" if r["is_non_sarcastic"] else "FAIL"
        print(f"  [{status}] (in={r['input_sarc_prob']:.2f}, out={r['output_sarc_prob']:.2f})")
        print(f"    IN:  {r['input']}")
        print(f"    OUT: {r['output']}")
        print()


if __name__ == "__main__":
    main()
