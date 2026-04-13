"""Generate outputs from all models on test.jsonl and save per-model CSVs.

Usage:
    python scripts/generate_test_outputs.py
    python scripts/generate_test_outputs.py --model llama-3.2-1b-instruct-context
    python scripts/generate_test_outputs.py --model llama-3.2-1b-instruct-context --with_context
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
ARTICLE_CACHE = PROJECT_ROOT / "data" / "processed" / "intermediate" / "article_scrape_cache.jsonl"
CE_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "sar_to_non_context_enhanced"

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
    "llama-3.2-1b-instruct-context": PROJECT_ROOT / "outputs" / "llama-3.2-1b-instruct-context" / "sar-to-non" / "final",
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


def load_headline_to_body() -> dict[str, str]:
    """Build a lowercased-headline -> article_body lookup.

    Joins CE splits (which map headlines -> article_link) with the scrape cache
    (which maps article_link -> body_text). Returns empty string for headlines
    where no body is available.
    """
    url_to_body: dict[str, str] = {}
    if ARTICLE_CACHE.exists():
        with open(ARTICLE_CACHE) as f:
            for line in f:
                r = json.loads(line)
                if r.get("has_body") and r.get("body_text"):
                    url_to_body[r["url"]] = r["body_text"]

    headline_to_body: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        path = CE_SPLITS_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                url = r.get("article_link", "")
                body = url_to_body.get(url, "")
                if body:
                    key = r["original_headline"].strip().lower()
                    headline_to_body[key] = body
    return headline_to_body


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


def build_llama_user_message(headline: str, body: str | None) -> str:
    if body:
        return (
            "Rewrite this sarcastic headline as a neutral, non-sarcastic news headline.\n\n"
            f"Headline: {headline}\n\n"
            f"Article context:\n{body}"
        )
    return f"Rewrite this sarcastic headline as a neutral, non-sarcastic news headline:\n\n{headline}"


def generate_llama(
    model_path: Path,
    inputs: list[str],
    bodies: list[str] | None = None,
    batch_size: int = 8,
    max_input_length: int = 1024,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16).to(device)
    model.eval()

    if bodies is None:
        bodies = [""] * len(inputs)

    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i : i + batch_size]
        batch_bodies = bodies[i : i + batch_size]
        prompts = []
        for inp, body in zip(batch_inputs, batch_bodies):
            messages = [
                {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": build_llama_user_message(inp, body or None)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length).to(device)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help=f"Generate for only this model. Options: {', '.join(MODELS.keys())}")
    parser.add_argument("--with_context", action="store_true",
                        help="Llama only: feed article body at inference when available. "
                             "Output CSV gets '-with-context' suffix.")
    args = parser.parse_args()

    records = load_test_data()
    inputs = [r["input"] for r in records]
    print(f"Loaded {len(records)} test examples")

    if args.model and args.model not in MODELS:
        raise ValueError(f"Unknown model {args.model!r}. Options: {list(MODELS.keys())}")
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS

    bodies: list[str] | None = None
    if args.with_context:
        print("\nLoading article bodies for in-context inference...")
        headline_to_body = load_headline_to_body()
        bodies = [headline_to_body.get(inp.strip().lower(), "") for inp in inputs]
        n_with_body = sum(1 for b in bodies if b)
        print(f"  Matched {n_with_body}/{len(inputs)} test examples to article bodies "
              f"({100 * n_with_body / len(inputs):.1f}%)")

    for model_name, model_path in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"Generating with {model_name} ({model_path})")
        print(f"{'='*60}")

        if "llama" in model_name:
            predictions = generate_llama(
                model_path, inputs, bodies=bodies if args.with_context else None,
            )
        else:
            if args.with_context:
                print("  [skip] --with_context only affects llama models; running headline-only")
            predictions = generate_seq2seq(model_path, inputs)

        suffix = "-with-context" if args.with_context and "llama" in model_name else ""
        save_csv(records, predictions, f"{model_name}{suffix}")

    print("\nDone! All outputs saved to generation_outputs/")


if __name__ == "__main__":
    main()
