"""Evaluation script for trained sarcasm style transfer models.

Loads a trained checkpoint and evaluates on the test set.
Computes BLEU, METEOR, and generates sample outputs.

Usage:
    python scripts/evaluate.py --checkpoint outputs/t5-base/sar-to-non/final \
        --direction sar-to-non

    # Custom test file
    python scripts/evaluate.py --checkpoint outputs/gpt2/sar-to-non/final \
        --direction sar-to-non --test_file data/splits/sar_to_non/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nltk
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQ2SEQ_MODELS = {"t5", "bart", "flan"}


def is_seq2seq(model_name: str) -> bool:
    return any(k in model_name.lower() for k in SEQ2SEQ_MODELS)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_test_path(direction: str) -> Path:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if direction == "sar-to-non":
        return splits_dir / "sar_to_non" / "test.jsonl"
    return splits_dir / "test.jsonl"


def prepare_examples(records: list[dict], direction: str, model_name: str) -> list[dict]:
    examples = []
    for r in records:
        if direction == "sar-to-non":
            source = r["original_headline"]
            target = r["generated_headline"]
            if is_seq2seq(model_name):
                input_text = f"desarcasm: {source}"
            else:
                input_text = source
        else:
            strategy = r["strategy"]
            source = r.get("non_sarcastic_source", r.get("original_headline", ""))
            target = r["generated_headline"]
            input_text = f"<{strategy}> {source}"

        examples.append({
            "input_text": input_text,
            "target_text": target,
            "strategy": r.get("strategy", "unknown"),
            "original": r.get("original_headline", source),
        })
    return examples


SEPARATOR = " → "


def generate_seq2seq(model, tokenizer, inputs: list[str], max_length: int, batch_size: int) -> list[str]:
    outputs = []
    device = model.device
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=max_length, num_beams=4)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs


def generate_causal(model, tokenizer, inputs: list[str], max_length: int, batch_size: int) -> list[str]:
    outputs = []
    device = model.device
    for i in range(0, len(inputs), batch_size):
        batch = [inp + SEPARATOR for inp in inputs[i : i + batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        prompt_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        # Decode only the generated portion
        for gen in generated:
            decoded = tokenizer.decode(gen[prompt_len:], skip_special_tokens=True).strip()
            outputs.append(decoded)
    return outputs


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    import evaluate

    results = {}

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    results["bleu"] = round(bleu_result["bleu"], 4)

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=predictions, references=references)
    results["meteor"] = round(meteor_result["meteor"], 4)

    # ROUGE-L
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    results["rouge_l"] = round(rouge_result["rougeL"], 4)

    return results


def compute_metrics_by_strategy(examples: list[dict], predictions: list[str]) -> dict:
    """Compute metrics grouped by sarcasm strategy."""
    from collections import defaultdict
    import evaluate

    by_strategy = defaultdict(lambda: {"preds": [], "refs": []})
    for ex, pred in zip(examples, predictions):
        s = ex["strategy"]
        by_strategy[s]["preds"].append(pred)
        by_strategy[s]["refs"].append(ex["target_text"])

    bleu = evaluate.load("bleu")
    results = {}
    for strategy, data in sorted(by_strategy.items()):
        b = bleu.compute(predictions=data["preds"], references=[[r] for r in data["refs"]])
        results[strategy] = {"bleu": round(b["bleu"], 4), "count": len(data["preds"])}

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate sarcasm style transfer model")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    p.add_argument("--direction", type=str, required=True, choices=["sar-to-non", "non-to-sar"])
    p.add_argument("--test_file", type=str, default=None, help="Override test file path")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=50, help="Number of sample outputs to save")
    p.add_argument("--output_dir", type=str, default=None, help="Where to save results (default: checkpoint parent)")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect model type from checkpoint config
    config_path = checkpoint / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("model_type", "")
    seq2seq = is_seq2seq(model_type)

    print(f"Checkpoint: {checkpoint}")
    print(f"Model type: {model_type} ({'seq2seq' if seq2seq else 'causal'})")
    print(f"Direction:  {args.direction}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint)).to(device)
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint)).to(device)

    model.eval()

    # Load test data
    test_path = Path(args.test_file) if args.test_file else get_test_path(args.direction)
    raw_records = load_jsonl(test_path)
    examples = prepare_examples(raw_records, args.direction, model_type)
    print(f"Test set: {len(examples)} examples from {test_path.name}")

    # Generate predictions
    input_texts = [ex["input_text"] for ex in examples]
    print("Generating predictions...")

    if seq2seq:
        predictions = generate_seq2seq(model, tokenizer, input_texts, args.max_length, args.batch_size)
    else:
        predictions = generate_causal(model, tokenizer, input_texts, args.max_length, args.batch_size)

    references = [ex["target_text"] for ex in examples]

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    strategy_metrics = compute_metrics_by_strategy(examples, predictions)

    results = {
        "checkpoint": str(checkpoint),
        "direction": args.direction,
        "test_file": str(test_path),
        "num_examples": len(examples),
        "metrics": metrics,
        "metrics_by_strategy": strategy_metrics,
    }

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"  BLEU:    {metrics['bleu']}")
    print(f"  METEOR:  {metrics['meteor']}")
    print(f"  ROUGE-L: {metrics['rouge_l']}")

    # Save sample outputs
    samples = []
    for i in range(min(args.num_samples, len(examples))):
        samples.append({
            "input": examples[i]["original"],
            "strategy": examples[i]["strategy"],
            "reference": references[i],
            "prediction": predictions[i],
        })

    samples_path = output_dir / "sample_outputs.json"
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Samples saved to {samples_path}")

    # Print strategy breakdown
    print("\nBy strategy:")
    for strategy, m in strategy_metrics.items():
        print(f"  {strategy:25s}  BLEU={m['bleu']:.4f}  (n={m['count']})")


if __name__ == "__main__":
    main()
