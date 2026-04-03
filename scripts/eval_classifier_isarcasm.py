"""Evaluate sarcasm classifiers on two benchmarks:
  1. NHDSD (News Headlines Dataset for Sarcasm Detection) — news domain
  2. iSarcasmEval Task A English test set — tweet domain

Run with one or more models to compare cross-domain generalisation.

Usage:
    # Our classifier only
    uv run python scripts/eval_classifier_isarcasm.py

    # Add a tweet-trained baseline
    uv run python scripts/eval_classifier_isarcasm.py \
        --models loyongzhe/sarcasm-classifier-binary \
                 cardiffnlp/twitter-roberta-base-irony

    # Any HuggingFace model
    uv run python scripts/eval_classifier_isarcasm.py --models <hf-model-id> ...
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NHDSD_PATH = PROJECT_ROOT / "data" / "raw" / "Sarcasm_Headlines_Dataset_v2.json"
ISARCASM_TEST_URL = (
    "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv"
)
BATCH_SIZE = 32
DEFAULT_MODELS = ["loyongzhe/sarcasm-classifier-binary"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_nhdsd(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["headline"])
            labels.append(int(r["is_sarcastic"]))
    return texts, labels


def load_isarcasm_test() -> tuple[list[str], list[int]]:
    import csv
    print(f"Downloading iSarcasmEval test set...")
    with urllib.request.urlopen(ISARCASM_TEST_URL) as resp:
        content = resp.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(content)))

    text_col = next((c for c in rows[0] if c.lower() in ("tweet", "text", "sentence")), None)
    label_col = next((c for c in rows[0] if c.lower() in ("sarcastic", "label", "class")), None)
    if not text_col or not label_col:
        raise ValueError(f"Unexpected columns: {list(rows[0].keys())}")

    texts = [r[text_col] for r in rows]
    labels = [int(r[label_col]) for r in rows]
    return texts, labels


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def find_sarcastic_idx(model) -> int:
    id2label = model.config.id2label
    for idx, label in id2label.items():
        ll = label.lower().replace(" ", "_")
        if ll in ("sarcastic", "sarc", "1") or ("sarcastic" in ll and "non" not in ll):
            return int(idx)
    return 1


def run_classifier(
    model_name: str,
    texts: list[str],
    device: str,
) -> tuple[list[int], list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    sarc_idx = find_sarcastic_idx(model)

    pred_labels, pred_probs = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        for p in probs:
            sp = p[sarc_idx].item()
            pred_probs.append(round(sp, 4))
            pred_labels.append(1 if sp > 0.5 else 0)
        print(f"    {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r", flush=True)

    print()
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return pred_labels, pred_probs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def evaluate(gold: list[int], pred: list[int], dataset_name: str, model_name: str):
    macro_f1 = f1_score(gold, pred, average="macro")
    acc = accuracy_score(gold, pred)
    report = classification_report(
        gold, pred, target_names=["non-sarcastic", "sarcastic"], digits=4
    )
    pos = sum(gold)
    print(f"\n{'─'*60}")
    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset_name}  (N={len(gold)}, sarcastic={pos}, non-sarc={len(gold)-pos})")
    print(f"Macro F1:  {macro_f1:.4f}    Accuracy: {acc:.4f}")
    print(report)
    return macro_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="One or more HuggingFace model IDs to evaluate",
    )
    parser.add_argument("--skip_nhdsd", action="store_true")
    parser.add_argument("--skip_isarcasm", action="store_true")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load datasets once
    datasets = {}
    if not args.skip_nhdsd:
        print("Loading NHDSD...")
        nhdsd_texts, nhdsd_labels = load_nhdsd(NHDSD_PATH)
        datasets["NHDSD (news headlines)"] = (nhdsd_texts, nhdsd_labels)
        print(f"  {len(nhdsd_texts)} examples")

    if not args.skip_isarcasm:
        isarcasm_texts, isarcasm_labels = load_isarcasm_test()
        datasets["iSarcasmEval Task A (tweets)"] = (isarcasm_texts, isarcasm_labels)
        print(f"  {len(isarcasm_texts)} examples")

    # Summary table
    summary: list[tuple[str, str, float]] = []

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        for dataset_name, (texts, labels) in datasets.items():
            print(f"  → {dataset_name}...")
            pred_labels, _ = run_classifier(model_name, texts, device)
            macro_f1 = evaluate(labels, pred_labels, dataset_name, model_name)
            summary.append((model_name, dataset_name, macro_f1))

    # Cross-model summary table
    if len(args.models) > 1 or len(datasets) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY — Macro F1")
        print(f"{'Model':<50} {'Dataset':<35} {'Macro F1':>8}")
        print("─" * 95)
        for model_name, dataset_name, f1 in summary:
            short = model_name.split("/")[-1][:48]
            print(f"{short:<50} {dataset_name:<35} {f1:>8.4f}")


if __name__ == "__main__":
    main()
