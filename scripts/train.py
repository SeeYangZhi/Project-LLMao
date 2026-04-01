"""Unified training script for sarcasm style transfer models.

Supports T5-base, GPT-2, and BART on both transfer directions.

Usage:
    # Sarcastic → Non-sarcastic (primary)
    python scripts/train.py --model t5-base --direction sar-to-non

    # Non-sarcastic → Sarcastic (secondary, with strategy control codes)
    python scripts/train.py --model gpt2 --direction non-to-sar

    # Override hyperparameters
    python scripts/train.py --model facebook/bart-base --direction sar-to-non \
        --lr 5e-5 --batch_size 32 --epochs 5

    # Quick test run
    python scripts/train.py --model t5-base --direction sar-to-non --max_steps 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nltk
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

nltk.download("punkt_tab", quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model type detection
SEQ2SEQ_MODELS = {"t5", "bart", "flan"}
CAUSAL_MODELS = {"gpt2", "gpt"}


def is_seq2seq(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(k in name_lower for k in SEQ2SEQ_MODELS)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_data_paths(direction: str) -> dict[str, Path]:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if direction == "sar-to-non":
        base = splits_dir / "sar_to_non"
    else:
        base = splits_dir
    return {
        "train": base / "train.jsonl",
        "val": base / "val.jsonl",
        "test": base / "test.jsonl",
    }


def prepare_examples(records: list[dict], direction: str, model_name: str) -> list[dict]:
    """Map raw JSONL records to input_text / target_text pairs."""
    examples = []
    for r in records:
        if direction == "sar-to-non":
            # Input: sarcastic headline → Output: non-sarcastic
            source = r["original_headline"]
            target = r["generated_headline"]
            # T5 needs a task prefix (pretrained with prefixes); BART and GPT-2 do not
            if "t5" in model_name.lower() or "flan" in model_name.lower():
                input_text = f"desarcasm: {source}"
            else:
                input_text = source
        else:
            # Input: non-sarcastic + strategy → Output: sarcastic
            strategy = r["strategy"]
            source = r.get("non_sarcastic_source", r.get("original_headline", ""))
            target = r["generated_headline"]
            if is_seq2seq(model_name):
                input_text = f"<{strategy}> {source}"
            else:
                input_text = f"<{strategy}> {source}"

        examples.append({"input_text": input_text, "target_text": target})
    return examples


# ---------------------------------------------------------------------------
# Seq2Seq tokenization (T5, BART)
# ---------------------------------------------------------------------------
def tokenize_seq2seq(examples, tokenizer, max_length):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ---------------------------------------------------------------------------
# Causal LM tokenization (GPT-2)
# ---------------------------------------------------------------------------
SEPARATOR = " → "


def tokenize_causal(examples, tokenizer, max_length):
    """Concatenate input + separator + target. Mask loss on input tokens."""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for inp, tgt in zip(examples["input_text"], examples["target_text"]):
        prompt = inp + SEPARATOR
        full_text = prompt + tgt + tokenizer.eos_token

        full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length)

        input_ids = full_enc["input_ids"]
        prompt_len = len(prompt_enc["input_ids"])

        # Mask the prompt portion so loss is only on the target
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(full_enc["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def build_compute_metrics(tokenizer, seq2seq: bool):
    import evaluate

    bleu_metric = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if seq2seq:
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        else:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Filter out empty predictions
        pairs = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p and l]
        if not pairs:
            return {"bleu": 0.0}

        preds_filtered, labels_filtered = zip(*pairs)
        result = bleu_metric.compute(
            predictions=list(preds_filtered),
            references=[[l] for l in labels_filtered],
        )
        return {"bleu": round(result["bleu"], 4)}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def reverse_non_to_sar_records(split: str, seed: int) -> list[dict]:
    """Load non-to-sar split, pick one random variant per source, reverse the pair.

    Each source has 6 strategy variants all mapping to the same non_sarcastic_source.
    We pick one random variant per source to avoid duplicating targets.
    Returns records in sar-to-non format: original_headline=sarcastic, generated_headline=non-sarcastic.
    """
    import random

    rng = random.Random(seed)
    path = PROJECT_ROOT / "data" / "splits" / f"{split}.jsonl"
    records = load_jsonl(path)

    # Group by non_sarcastic_source, pick one random variant
    by_source: dict[str, list[dict]] = {}
    for r in records:
        key = r["non_sarcastic_source"]
        by_source.setdefault(key, []).append(r)

    reversed_records = []
    for source, variants in by_source.items():
        chosen = rng.choice(variants)
        reversed_records.append({
            "original_headline": chosen["generated_headline"],  # sarcastic → input
            "generated_headline": chosen["non_sarcastic_source"],  # non-sarcastic → target
            "strategy": chosen["strategy"],
            "type": "sarcastic_to_non",
        })

    return reversed_records


def parse_args():
    p = argparse.ArgumentParser(description="Train sarcasm style transfer model")
    p.add_argument("--model", type=str, default="t5-base",
                    help="HuggingFace model name (t5-base, gpt2, facebook/bart-base)")
    p.add_argument("--direction", type=str, default="sar-to-non",
                    choices=["sar-to-non", "non-to-sar"],
                    help="Transfer direction")
    p.add_argument("--augment_reversed", action="store_true",
                    help="For sar-to-non: also include reversed non-to-sar pairs (1 per source)")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    p.add_argument("--max_steps", type=int, default=-1,
                    help="Max training steps (overrides epochs, useful for testing)")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: outputs/{model}/{direction})")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Determine output directory
    model_short = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "outputs" / model_short / args.direction)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reporting
    report_to = "wandb" if args.wandb else "none"

    print(f"Model:     {args.model}")
    print(f"Direction: {args.direction}")
    print(f"Output:    {output_dir}")
    print(f"LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}, MaxLen: {args.max_length}")
    print()

    # Load data
    data_paths = get_data_paths(args.direction)
    train_records = load_jsonl(data_paths["train"])
    val_records = load_jsonl(data_paths["val"])

    # Optionally augment sar-to-non with reversed non-to-sar pairs
    if args.augment_reversed and args.direction == "sar-to-non":
        reversed_train = reverse_non_to_sar_records("train", args.seed)
        reversed_val = reverse_non_to_sar_records("val", args.seed)
        print(f"Augmenting with {len(reversed_train)} reversed train + {len(reversed_val)} reversed val pairs")
        train_records.extend(reversed_train)
        val_records.extend(reversed_val)

    raw_train = prepare_examples(train_records, args.direction, args.model)
    raw_val = prepare_examples(val_records, args.direction, args.model)

    train_ds = Dataset.from_list(raw_train)
    val_ds = Dataset.from_list(raw_val)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Load tokenizer and model
    seq2seq = is_seq2seq(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

        tok_fn = lambda ex: tokenize_seq2seq(ex, tokenizer, args.max_length)
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["input_text", "target_text"])
        val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["input_text", "target_text"])

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=500,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=args.max_length,
            logging_steps=100,
            report_to=report_to,
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=build_compute_metrics(tokenizer, seq2seq=True),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    else:
        # GPT-2 style causal LM
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.resize_token_embeddings(len(tokenizer))

        tok_fn = lambda ex: tokenize_causal(ex, tokenizer, args.max_length)
        train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["input_text", "target_text"])
        val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["input_text", "target_text"])

        def causal_data_collator(features):
            """Pad input_ids, labels, and attention_mask to the same length."""
            max_len = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "labels": [], "attention_mask": []}
            for f in features:
                pad_len = max_len - len(f["input_ids"])
                batch["input_ids"].append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
                batch["labels"].append(f["labels"] + [-100] * pad_len)
                batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            return {k: torch.tensor(v) for k, v in batch.items()}

        data_collator = causal_data_collator

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=500,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=100,
            report_to=report_to,
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to {final_dir}")

    # Save training args for reproducibility
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
