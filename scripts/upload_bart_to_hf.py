"""Upload BART variants (base, CE, RL, CE+RL) to Hugging Face Hub.

Each variant goes to its own repo under SeeYangZhi/. Uses HUGGINGFACE_API_KEY
from the .env file. Disables Xet storage to avoid the known token-refresh bug
on very large uploads.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    {
        "name": "BART-Base-Sarcasm-Rewriter",
        "folder": PROJECT_ROOT / "checkpoints" / "bart-base" / "sar-to-non" / "final",
        "title": "BART-Base (Baseline)",
        "description": (
            "Baseline BART-base supervised fine-tuning on sarcastic->non-sarcastic "
            "headline pairs. No context enhancement, no RL."
        ),
        "training_note": "Standard cross-entropy on (sarcastic, non-sarcastic) pairs derived from NHDSD.",
    },
    {
        "name": "BART-Base-CE-Sarcasm-Rewriter",
        "folder": PROJECT_ROOT / "outputs" / "bart-base-ce" / "sar-to-non" / "final",
        "title": "BART-Base-CE (Context Enhanced)",
        "description": (
            "BART-base fine-tuned with **Context Enhancement**: during training, "
            "the article body is prepended to the sarcastic headline so the model "
            "can ground its rewrite in factual context."
        ),
        "training_note": (
            "Input format: `rewrite to non-sarcastic: <article_body> [SEP] <sarcastic_headline>`. "
            "The context provides disambiguation for headlines whose sarcasm relies on world knowledge."
        ),
    },
    {
        "name": "BART-Base-RL-Sarcasm-Rewriter",
        "folder": PROJECT_ROOT / "outputs" / "bart-base-rl" / "sar-to-non" / "best",
        "title": "BART-Base-RL (REINFORCE)",
        "description": (
            "BART-base further trained with **REINFORCE + KL penalty** on top of "
            "the supervised baseline. The reward encourages high semantic similarity "
            "combined with a low irony-classifier score on the output."
        ),
        "training_note": (
            "REINFORCE policy-gradient fine-tuning with KL divergence against the SFT model "
            "to prevent drift. Reward = similarity * (1 - sarcasm_prob)."
        ),
    },
    {
        "name": "BART-Base-CE-RL-Sarcasm-Rewriter",
        "folder": PROJECT_ROOT / "outputs" / "bart-base-ce-rl" / "sar-to-non" / "best",
        "title": "BART-Base-CE+RL (Best Variant)",
        "description": (
            "**Our best BART variant.** Combines Context Enhancement supervised fine-tuning "
            "with a subsequent REINFORCE + KL pass. Used as the default model in the "
            "Project LLMao webapp playground."
        ),
        "training_note": (
            "Stage 1: SFT with article context (`BART-Base-CE`). Stage 2: REINFORCE with "
            "KL penalty against the SFT model. Reward = similarity * (1 - sarcasm_prob)."
        ),
    },
]


def build_readme(variant: dict) -> str:
    return f"""---
license: mit
base_model: facebook/bart-base
tags:
- text2text-generation
- style-transfer
- sarcasm
- bart
- seq2seq
language:
- en
pipeline_tag: summarization
---

# {variant['title']}

{variant['description']}

Part of the **Project LLMao** sarcasm style transfer suite (CS4248 Team 14, NUS AY2025/26 S2).
This model rewrites sarcastic news headlines as neutral, factual equivalents while
preserving the underlying meaning.

## Task

**Input**: A sarcastic news headline
**Output**: A non-sarcastic rewrite

Example:
- In: *"Area Man Passionate Defender Of What He Imagines Constitution To Be"*
- Out: *"Man defends his interpretation of the Constitution."*

## Training

- **Base model**: [`facebook/bart-base`](https://huggingface.co/facebook/bart-base) (139M params)
- **Method**: {variant['training_note']}
- **Dataset**: 71,730 sarcastic / non-sarcastic headline pairs derived from NHDSD
  (News Headlines Dataset for Sarcasm Detection), augmented with 6 sarcasm strategy
  variants (sarcasm, irony, satire, overstatement, understatement, rhetorical question).
- **Input prefix**: `rewrite to non-sarcastic: ` is prepended to every input at inference time.
- **Generation**: beam search with `num_beams=4`, `max_length=128`.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "SeeYangZhi/{variant['name']}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

headline = "Area Man Passionate Defender Of What He Imagines Constitution To Be"
prompt = "rewrite to non-sarcastic: " + headline

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
outputs = model.generate(**inputs, max_length=128, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation

Evaluated on a 2,857-sample held-out test split alongside 13 other model variants
(BART, T5 baselines, LLaMA 3.2, ablation studies). Metrics include:

| Metric | Direction |
|---|---|
| Hard Flip Rate (% of samples where sarcasm was removed) | higher ↑ |
| Semantic Similarity (all-MiniLM-L6-v2 cosine) | higher ↑ |
| BLEU vs input (lower = more genuine rewriting) | lower ↓ |
| Perplexity (GPT-2) | lower ↓ |
| Normalized edit distance | higher ↑ |
| Paraphrase score (low = real rewriting) | lower ↓ |

Full per-variant numbers are published alongside the Project LLMao webapp.

## Related models

- [`SeeYangZhi/Llama-3.2-1B-Sarcasm-Rewriter`](https://huggingface.co/SeeYangZhi/Llama-3.2-1B-Sarcasm-Rewriter) — instruction-tuned LLaMA variant
- `SeeYangZhi/BART-Base-Sarcasm-Rewriter` — supervised baseline
- `SeeYangZhi/BART-Base-CE-Sarcasm-Rewriter` — context-enhanced SFT
- `SeeYangZhi/BART-Base-RL-Sarcasm-Rewriter` — REINFORCE on top of baseline
- `SeeYangZhi/BART-Base-CE-RL-Sarcasm-Rewriter` — CE + RL (best)

## License

MIT, inheriting from `facebook/bart-base`. The NHDSD dataset is used under its
original research-use terms.
"""


def load_token() -> str:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HUGGINGFACE_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_KEY not found in .env or environment")
    return token


def main():
    token = load_token()
    api = HfApi(token=token)

    for variant in VARIANTS:
        repo_id = f"SeeYangZhi/{variant['name']}"
        folder = variant["folder"]

        if not folder.exists():
            print(f"[skip] {repo_id}: folder missing at {folder}")
            continue

        # Write README directly into the model folder
        readme_path = folder / "README.md"
        readme_path.write_text(build_readme(variant))

        print(f"\n=== {repo_id} ===")
        print(f"  Source: {folder}")
        print(f"  Creating/updating repo...")
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

        print(f"  Uploading folder...")
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {variant['title']} for sarcasm rewriting",
            ignore_patterns=["training_args.bin", "checkpoint-*", "*.pt"],
        )
        print(f"  Done -> https://huggingface.co/{repo_id}")

    print("\nAll BART variants uploaded.")


if __name__ == "__main__":
    main()
