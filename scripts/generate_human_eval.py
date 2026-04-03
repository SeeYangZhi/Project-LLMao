"""Generate non-sarcastic outputs from multiple BART checkpoints for human evaluation.

Produces a CSV with original sarcastic headline + outputs from each model,
ready for human grading.

Usage:
    uv run python scripts/generate_human_eval.py
"""

from __future__ import annotations

import argparse
import csv
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


def generate_outputs(model, tokenizer, texts: list[str], device: str) -> list[str]:
    all_outputs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                length_penalty=1.0,
            )
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)
        print(f"    Generated {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", flush=True)
    return all_outputs


def classify_batch(model, tokenizer, texts: list[str], sarcastic_idx: int, device: str) -> list[float]:
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        for p in probs:
            all_probs.append(round(p[sarcastic_idx].item(), 4))
    return all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "human_eval.csv"))
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Define models to compare
    models = {
        "BART_RL": PROJECT_ROOT / "outputs" / "bart-base-rl" / "sar-to-non" / "best",
        "BART_CE_RL": PROJECT_ROOT / "outputs" / "bart-base-ce-rl" / "sar-to-non" / "best",
    }

    # Load classifier
    print(f"Loading classifier {CLASSIFIER_MODEL}...")
    cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    cls_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL).to(device)
    cls_model.eval()

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

    # Load headlines
    headlines = load_sarcastic_headlines(HEADLINES_PATH)
    input_texts = [h["headline"] for h in headlines]
    print(f"Loaded {len(headlines)} sarcastic Onion headlines\n")

    # Generate outputs from each model
    all_outputs = {}
    all_sarc_probs = {}
    for name, ckpt_path in models.items():
        print(f"Loading {name} from {ckpt_path}...")
        tok = BartTokenizer.from_pretrained(ckpt_path)
        mdl = BartForConditionalGeneration.from_pretrained(ckpt_path).to(device)
        mdl.eval()

        print(f"  Generating...")
        outputs = generate_outputs(mdl, tok, input_texts, device)
        all_outputs[name] = outputs

        print(f"  Classifying outputs...")
        probs = classify_batch(cls_model, cls_tokenizer, outputs, sarcastic_idx, device)
        all_sarc_probs[name] = probs

        # Free memory
        del mdl, tok
        if device == "cuda":
            torch.cuda.empty_cache()
        print()

    # Classify original inputs
    print("Classifying original inputs...")
    input_sarc_probs = classify_batch(cls_model, cls_tokenizer, input_texts, sarcastic_idx, device)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(models.keys())
    fieldnames = [
        "id",
        "sarcastic_input",
        "input_sarc_prob",
    ]
    for name in model_names:
        fieldnames.append(f"{name}_output")
        fieldnames.append(f"{name}_sarc_prob")
    fieldnames.extend([
        "human_grade_RL",
        "human_grade_CE_RL",
    ])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, headline in enumerate(input_texts):
            row = {
                "id": i + 1,
                "sarcastic_input": headline,
                "input_sarc_prob": input_sarc_probs[i],
            }
            for name in model_names:
                row[f"{name}_output"] = all_outputs[name][i]
                row[f"{name}_sarc_prob"] = all_sarc_probs[name][i]
            row["human_grade_RL"] = ""
            row["human_grade_CE_RL"] = ""
            writer.writerow(row)

    print(f"Wrote {len(input_texts)} rows to {output_path}")

    # Print summary stats
    for name in model_names:
        probs = all_sarc_probs[name]
        desarc = sum(1 for p in probs if p <= 0.5)
        avg_prob = sum(probs) / len(probs)
        identical = sum(
            1 for inp, out in zip(input_texts, all_outputs[name])
            if inp.strip().lower() == out.strip().lower()
        )
        print(f"\n{name}:")
        print(f"  De-sarcasm rate: {desarc}/{len(probs)} ({100*desarc/len(probs):.1f}%)")
        print(f"  Avg sarcasm prob: {avg_prob:.4f}")
        print(f"  Identical to input: {identical} ({100*identical/len(probs):.1f}%)")


if __name__ == "__main__":
    main()
