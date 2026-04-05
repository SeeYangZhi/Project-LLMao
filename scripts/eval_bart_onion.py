"""Evaluate a sar-to-non model on sarcastic Onion headlines.

Supports BART (seq2seq) and LLaMA/causal models. Loads sarcastic headlines
from onion_headlines_classified.jsonl, generates non-sarcastic versions,
then classifies the outputs to check if they are actually non-sarcastic.

Usage:
    # BART
    uv run python scripts/eval_bart_onion.py --model outputs/bart-base-rl/sar-to-non/best

    # LLaMA
    uv run python scripts/eval_bart_onion.py --model outputs/llama-3.2-1b-instruct/sar-to-non/final
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADLINES_PATH = PROJECT_ROOT / "data" / "processed" / "onion_headlines_classified.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "bart-base-rl" / "sar-to-non" / "best"
CLASSIFIER_MODEL = "loyongzhe/sarcasm-classifier-binary"
BATCH_SIZE = 32

SYSTEM_PROMPT = (
    "You are a writing assistant. Rewrite sarcastic news headlines as neutral, "
    "factual equivalents that preserve the core meaning without irony or mockery. "
    "Respond with only the rewritten headline, no explanation."
)


def load_sarcastic_headlines(path: Path) -> list[dict]:
    headlines = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["sarcasm_probability"] > 0.5:
                headlines.append(rec)
    return headlines


def is_seq2seq_model(model_path: str) -> bool:
    config = AutoConfig.from_pretrained(model_path)
    return config.is_encoder_decoder


def generate_seq2seq(model, tokenizer, texts: list[str], device: str) -> list[str]:
    all_outputs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_length=128, num_beams=4, length_penalty=1.0,
            )
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)
        print(f"  Generated {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", flush=True)
    return all_outputs


def generate_causal(model, tokenizer, texts: list[str], device: str) -> list[str]:
    all_outputs = []
    # Process one at a time since each prompt has different length
    for i, text in enumerate(texts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Rewrite this sarcastic headline as a neutral, non-sarcastic news headline:\n\n{text}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
            )
        # Decode only the new tokens
        output_ids = gen_ids[0][prompt_len:]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        all_outputs.append(decoded)

        if (i + 1) % 50 == 0 or i + 1 == len(texts):
            print(f"  Generated {i + 1}/{len(texts)}", flush=True)
    return all_outputs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Model path or HF name")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Auto-detect model type
    model_path = args.model
    seq2seq = is_seq2seq_model(model_path)
    model_type = "seq2seq" if seq2seq else "causal"
    print(f"Loading {model_type} model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
        ).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.eval()

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
    input_texts = [h["headline"] for h in headlines]
    print(f"Loaded {len(headlines)} sarcastic Onion headlines\n")

    # Generate
    print("Generating outputs...")
    if seq2seq:
        generated = generate_seq2seq(model, tokenizer, input_texts, device)
    else:
        generated = generate_causal(model, tokenizer, input_texts, device)

    # Classify outputs
    print("\nClassifying outputs...")
    results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch_gen = generated[i : i + BATCH_SIZE]
        batch_h = headlines[i : i + BATCH_SIZE]

        cls_inputs = cls_tokenizer(
            batch_gen, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = cls_model(**cls_inputs).logits
            probs = torch.softmax(logits, dim=-1)

        for h, gen_text, prob in zip(batch_h, batch_gen, probs):
            sarc_prob = prob[sarcastic_idx].item()
            results.append({
                "input": h["headline"],
                "input_sarc_prob": h["sarcasm_probability"],
                "output": gen_text,
                "output_sarc_prob": round(sarc_prob, 4),
                "is_non_sarcastic": sarc_prob <= 0.5,
            })

    # Statistics
    total = len(results)
    success = sum(1 for r in results if r["is_non_sarcastic"])
    avg_output_sarc = sum(r["output_sarc_prob"] for r in results) / total
    identical = sum(1 for r in results if r["input"].strip().lower() == r["output"].strip().lower())

    print(f"\n{'='*60}")
    print(f"Model:                         {model_path}")
    print(f"Type:                          {model_type}")
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
