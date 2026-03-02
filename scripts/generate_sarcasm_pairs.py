#!/usr/bin/env python3
"""
Generate sarcasm pairs using MiniMax M2.5 via OpenCode Zen API.

Usage:
    uv run python scripts/generate_sarcasm_pairs.py
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("opencode-zen-apikey")
if not OPENAI_API_KEY:
    raise ValueError("opencode-zen-apikey not found in .env file")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://opencode.ai/zen/v1",
)

SYSTEM_PROMPT = """You are a sarcasm style transfer expert. Output ONLY JSON."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": 'Sarcastic: "Great job on the update, everything is broken now"\nNon-sarcastic:',
    },
    {
        "role": "assistant",
        "content": '{"non_sarcastic": "The latest update has caused multiple issues", "strategy": "sarcasm"}',
    },
    {
        "role": "user",
        "content": 'Sarcastic: "I love waiting in line at the DMV for hours"\nNon-sarcastic:',
    },
    {
        "role": "assistant",
        "content": '{"non_sarcastic": "Waiting at the DMV takes several hours", "strategy": "irony"}',
    },
]


def load_dataset(path: str) -> list[dict]:
    """Load NHDSD dataset from JSONL format."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_opposite(headline: str, direction: str = "sarcastic_to_non") -> dict:
    """Generate opposite style headline using MiniMax."""

    if direction == "sarcastic_to_non":
        user_content = f'Sarcastic: "{headline}"\nNon-sarcastic:'
    else:
        user_content = f'Non-sarcastic: "{headline}"\nSarcastic:'

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model="minimax-m2.5-free",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
    )

    result_text = response.choices[0].message.content
    if not result_text:
        reasoning = response.choices[0].message.reasoning
        if reasoning:
            json_match = re.search(r"\{[^{}]*\}", reasoning)
            if json_match:
                result_text = json_match.group()

    if not result_text:
        print("Empty response")
        return None

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[^{}]*\}", result_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            print(f"Failed to parse: {result_text[:100]}...")
            return None

    return result


def main():
    # Paths
    dataset_path = Path("Sarcasm_Headlines_Dataset_v2.json")
    output_path = Path("data/processed/sarcasm_pairs_minimax.jsonl")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)

    # Separate sarcastic and non-sarcastic
    sarcastic_headlines = [d for d in data if d.get("is_sarcastic") == 1]
    non_sarcastic_headlines = [d for d in data if d.get("is_sarcastic") == 0]

    print(f"Total: {len(data)}")
    print(f"Sarcastic: {len(sarcastic_headlines)}")
    print(f"Non-sarcastic: {len(non_sarcastic_headlines)}")

    # Generate pairs
    pairs = []

    # Sarcastic → Non-sarcastic
    print(
        f"\nGenerating sarcastic → non-sarcastic ({len(sarcastic_headlines)} pairs)..."
    )
    for i, item in enumerate(sarcastic_headlines):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(sarcastic_headlines)}")

        result = generate_opposite(item["headline"], "sarcastic_to_non")
        if result:
            pairs.append(
                {
                    "sarcastic": item["headline"],
                    "non_sarcastic": result.get("non_sarcastic", ""),
                    "strategy": result.get("strategy", ""),
                    "direction": "sarcastic_to_non",
                }
            )

    # Non-sarcastic → Sarcastic
    print(
        f"\nGenerating non-sarcastic → sarcastic ({len(non_sarcastic_headlines)} pairs)..."
    )
    for i, item in enumerate(non_sarcastic_headlines):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(non_sarcastic_headlines)}")

        result = generate_opposite(item["headline"], "non_to_sarcastic")
        if result:
            pairs.append(
                {
                    "sarcastic": result.get("sarcastic", ""),
                    "non_sarcastic": item["headline"],
                    "strategy": result.get("strategy", ""),
                    "direction": "non_to_sarcastic",
                }
            )

    # Save to JSONL
    print(f"\nSaving {len(pairs)} pairs to {output_path}...")
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Done! Generated {len(pairs)} pairs.")


if __name__ == "__main__":
    main()
