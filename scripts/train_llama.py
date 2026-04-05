"""LoRA fine-tuning of Llama-3.2-1B-Instruct for sarcasm-to-non-sarcastic style transfer.

Uses PEFT LoRA so only ~6M of the 1.24B parameters are trained.
Loss is computed only on the assistant response (target headline), not the prompt.

Usage:
    # On SLURM (A100)
    sbatch scripts/run_sft_llama.sh

    # Local quick test
    uv run python scripts/train_llama.py --max_steps 20 --batch_size 2

    # Full run
    uv run python scripts/train_llama.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "splits" / "sar_to_non_context_enhanced"

SYSTEM_PROMPT = (
    "You are a writing assistant. Rewrite sarcastic news headlines as neutral, "
    "factual equivalents that preserve the core meaning without irony or mockery. "
    "Respond with only the rewritten headline, no explanation."
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_prompt(headline: str) -> str:
    return f"Rewrite this sarcastic headline as a neutral, non-sarcastic news headline:\n\n{headline}"


def tokenize_example(record: dict, tokenizer, max_length: int) -> dict:
    sarcastic = record["original_headline"]
    target = record["generated_headline"]

    # Render the prompt portion only (up to where assistant would respond)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(sarcastic)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + target + "<|eot_id|>"

    full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length)

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    prompt_len = len(prompt_enc["input_ids"])

    # Mask prompt tokens: loss only on the target headline
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(path: Path, tokenizer, max_length: int) -> Dataset:
    records = load_jsonl(path)
    tokenized = [tokenize_example(r, tokenizer, max_length) for r in records]
    return Dataset.from_list(tokenized)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class PaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_short = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "outputs" / model_short / "sar-to-non")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {args.model}")
    print(f"Data:     {args.data_dir}")
    print(f"Output:   {output_dir}")
    print(f"LoRA:     r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"LR: {args.lr}  Batch: {args.batch_size}  Grad accum: {args.grad_accum}  Epochs: {args.epochs}")
    print()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Datasets
    data_dir = Path(args.data_dir)
    print("Tokenizing train split...")
    train_ds = build_dataset(data_dir / "train.jsonl", tokenizer, args.max_length)
    print(f"  {len(train_ds)} examples")
    print("Tokenizing val split...")
    val_ds = build_dataset(data_dir / "val.jsonl", tokenizer, args.max_length)
    print(f"  {len(val_ds)} examples\n")

    # Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()  # required for gradient checkpointing with PEFT

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="wandb" if args.wandb else "none",
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=PaddingCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter only (small, ~50MB)
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved to {adapter_dir}")

    # Merge into full model for inference (same API as BART/T5 checkpoints)
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    final_dir = output_dir / "final"
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Merged model saved to {final_dir}")

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
