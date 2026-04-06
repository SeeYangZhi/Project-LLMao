"""Generate outputs from all models on test.jsonl and save per-model CSVs.

Usage:
    python scripts/generate_test_outputs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = PROJECT_ROOT / "test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "generation_outputs"

PREFIX = "rewrite to non-sarcastic: "

LLAMA_SYSTEM_PROMPT = (
    "You are a writing assistant. Rewrite sarcastic news headlines as neutral, "
    "factual equivalents that preserve the core meaning without irony or mockery. "
    "Respond with only the rewritten headline, no explanation."
)

MODELS = {
    "bart-base": PROJECT_ROOT / "checkpoints" / "bart-base" / "sar-to-non" / "final",
    "bart-base-ce": PROJECT_ROOT / "outputs" / "bart-base-ce" / "sar-to-non" / "final",
    "bart-base-ce-rl": PROJECT_ROOT / "outputs" / "bart-base-ce-rl" / "sar-to-non" / "best",
    "bart-base-rl": PROJECT_ROOT / "outputs" / "bart-base-rl" / "sar-to-non" / "best",
    "llama-3.2-1b-instruct": PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct" / "sar-to-non" / "final",
}


def load_test_data() -> list[dict]:
    records = []
    with open(TEST_FILE) as f:
        for line in f:
            r = json.loads(line)
            inp = r["input_text"]
            if inp.startswith(PREFIX):
                inp = inp[len(PREFIX):]
            records.append({"id": r["id"], "input": inp})
    return records


def generate_seq2seq(model_path: Path, inputs: list[str], batch_size: int = 32) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)
    model.eval()

    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=128, num_beams=4)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)
        if (i // batch_size) % 10 == 0:
            print(f"  batch {i // batch_size + 1}/{(len(inputs) + batch_size - 1) // batch_size}")
    return outputs


def generate_llama(model_path: Path, inputs: list[str], batch_size: int = 8) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16).to(device)
    model.eval()

    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i : i + batch_size]
        prompts = []
        for inp in batch_inputs:
            messages = [
                {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite this sarcastic headline as a neutral, non-sarcastic news headline:\n\n{inp}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        prompt_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        for gen in generated:
            decoded = tokenizer.decode(gen[prompt_len:], skip_special_tokens=True).strip()
            outputs.append(decoded)
        if (i // batch_size) % 10 == 0:
            print(f"  batch {i // batch_size + 1}/{(len(inputs) + batch_size - 1) // batch_size}")
    return outputs


def save_csv(records: list[dict], predictions: list[str], model_name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input", "output"])
        for rec, pred in zip(records, predictions):
            writer.writerow([rec["id"], rec["input"], pred])
    print(f"  Saved {path}")


def main():
    records = load_test_data()
    inputs = [r["input"] for r in records]
    print(f"Loaded {len(records)} test examples")

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Generating with {model_name} ({model_path})")
        print(f"{'='*60}")

        if "llama" in model_name:
            predictions = generate_llama(model_path, inputs)
        else:
            predictions = generate_seq2seq(model_path, inputs)

        save_csv(records, predictions, model_name)

    print("\nDone! All outputs saved to generation_outputs/")


if __name__ == "__main__":
    main()
