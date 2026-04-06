"""Generate LLaMA outputs on test.jsonl — one example at a time for MPS compatibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = PROJECT_ROOT / "test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "generation_outputs"

PREFIX = "rewrite to non-sarcastic: "
MODEL_PATH = PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct" / "sar-to-non" / "final"

SYSTEM_PROMPT = (
    "You are a writing assistant. Rewrite sarcastic news headlines as neutral, "
    "factual equivalents that preserve the core meaning without irony or mockery. "
    "Respond with only the rewritten headline, no explanation."
)


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


def main():
    records = load_test_data()
    print(f"Loaded {len(records)} test examples")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=torch.float32).to(device)
    model.eval()
    print("Model loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "llama-3.2-1b-instruct.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input", "output"])

        for i, rec in enumerate(records):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Rewrite this sarcastic headline as a neutral, non-sarcastic news headline:\n\n{rec['input']}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            prompt_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            decoded = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
            writer.writerow([rec["id"], rec["input"], decoded])

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(records)}")
                f.flush()

    print(f"Saved {path}")


if __name__ == "__main__":
    main()
