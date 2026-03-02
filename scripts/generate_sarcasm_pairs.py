#!/usr/bin/env python3
"""
Generate sarcasm pairs using MiniMax M2.5 via OpenCode Zen API.

Usage:
    uv run python scripts/generate_sarcasm_pairs.py
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env (supports rotation with opencode-zen-apikey, opencode-zen-apikey-0, -1, -2, etc.)
load_dotenv()

# Collect all available API keys
api_keys = []
for i in range(0, 100):  # Check for keys 0-99
    if i == 0:
        key_name = "opencode-zen-apikey"
    else:
        key_name = f"opencode-zen-apikey-{i}"
    key = os.getenv(key_name)
    if key:
        api_keys.append(key)

if not api_keys:
    raise ValueError("No opencode-zen-apikey found in .env file")

print(f"Found {len(api_keys)} API key(s)")

# Current key index
current_key_idx = 0


def get_client():
    """Get OpenAI client with current API key."""
    return OpenAI(
        api_key=api_keys[current_key_idx],
        base_url="https://opencode.ai/zen/v1",
    )


def rotate_key():
    """Rotate to next API key."""
    global current_key_idx
    current_key_idx = (current_key_idx + 1) % len(api_keys)
    print(f"  Rotated to API key {current_key_idx + 1}/{len(api_keys)}")


# Config
BATCH_SIZE = 50  # Save every N pairs
API_DELAY = 0.5  # Delay between requests to avoid rate limits
KEY_ROTATE_EVERY = 100  # Rotate key every N requests

SYSTEM_PROMPT = """You are a sarcasm style transfer expert.

Sarcasm Strategies (from iSarcasm dataset):
- sarcasm: Contradicts state of affairs, critical towards addressee
- irony: Contradicts state of affairs, not obviously critical
- satire: Appears to support, but contains mockery
- understatement: Undermines importance
- overstatement: Obviously exaggerated terms
- rhetorical_question: Question with inference contradicting reality

TASK TYPES:

1. Sarcastic → Non-sarcastic: Identify the sarcasm strategy used in the original headline, then convert to literal/non-sarcastic.
2. Non-sarcastic → Sarcastic: Apply a sarcasm strategy to create a sarcastic version.

Example 1 (Sarcastic → Non-sarcastic):
Input: Sarcastic: "Great job on the update, everything is broken now"
Output: {"non_sarcastic": "The latest update has caused multiple issues", "strategy_detected": "sarcasm"}

Example 2 (Sarcastic → Non-sarcastic):
Input: Sarcastic: "I love waiting in line at the DMV for hours"
Output: {"non_sarcastic": "Waiting at the DMV takes several hours", "strategy_detected": "irony"}

Example 3 (Non-sarcastic → Sarcastic):
Input: Non-sarcastic: "The update has caused multiple issues"
Output: {"sarcastic": "Great job on the update, everything is broken now", "strategy_applied": "sarcasm"}

Example 4 (Non-sarcastic → Sarcastic):
Input: Non-sarcastic: "Waiting at the DMV takes several hours"
Output: {"sarcastic": "I love waiting in line at the DMV for hours", "strategy_applied": "irony"}

Output format (JSON only):
- Sarcastic → Non-sarcastic: {"non_sarcastic": "...", "strategy_detected": "..."}
- Non-sarcastic → Sarcastic: {"sarcastic": "...", "strategy_applied": "..."}
No explanations or extra text."""


def load_dataset(path: str) -> list[dict]:
    """Load NHDSD dataset from JSONL format."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_existing_count(output_path: Path) -> int:
    """Load existing pair count for resume capability."""
    if output_path.exists():
        with open(output_path, "r") as f:
            return sum(1 for _ in f)
    return 0


def save_pairs(pairs: list[dict], output_path: Path, mode: str = "a"):
    """Save pairs to JSONL file."""
    with open(output_path, mode) as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


def generate_opposite(headline: str, direction: str = "sarcastic_to_non") -> dict:
    """Generate opposite style headline using MiniMax."""

    if direction == "sarcastic_to_non":
        user_content = f'Input: Sarcastic: "{headline}"\nOutput:'
    else:
        user_content = f'Input: Non-sarcastic: "{headline}"\nOutput:'

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    client = get_client()
    response = client.chat.completions.create(
        model="minimax-m2.5-free",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
        response_format={"type": "json_object"},  # Force JSON output
    )
    
    result_text = response.choices[0].message.content
    
    if not result_text:
        print("Empty response")
        return None
    
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return None
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

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)

    # Separate sarcastic and non-sarcastic
    sarcastic_headlines = [d for d in data if d.get("is_sarcastic") == 1]
    non_sarcastic_headlines = [d for d in data if d.get("is_sarcastic") == 0]

    print(f"Total: {len(data)}")
    print(f"Sarcastic: {len(sarcastic_headlines)}")
    print(f"Non-sarcastic: {len(non_sarcastic_headlines)}")

    # Check for resume
    existing_count = load_existing_count(output_path)
    if existing_count > 0:
        print(f"Found {existing_count} existing pairs, resuming...")

    # Generate pairs
    pairs = []
    total_generated = existing_count
    request_count = 0

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
                    "strategy_detected": result.get("strategy_detected", ""),
                    "direction": "sarcastic_to_non",
                }
            )

        request_count += 1
        if request_count % KEY_ROTATE_EVERY == 0:
            rotate_key()

        # Batch save
        if len(pairs) >= BATCH_SIZE:
            save_pairs(pairs, output_path)
            total_generated += len(pairs)
            print(f"    Saved {len(pairs)} pairs (total: {total_generated})")
            pairs = []

        time.sleep(API_DELAY)

    # Save remaining
    if pairs:
        save_pairs(pairs, output_path)
        total_generated += len(pairs)
        pairs = []

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
                    "strategy_applied": result.get("strategy_applied", ""),
                    "direction": "non_to_sarcastic",
                }
            )

        request_count += 1
        if request_count % KEY_ROTATE_EVERY == 0:
            rotate_key()

        # Batch save
        if len(pairs) >= BATCH_SIZE:
            save_pairs(pairs, output_path)
            total_generated += len(pairs)
            print(f"    Saved {len(pairs)} pairs (total: {total_generated})")
            pairs = []

        time.sleep(API_DELAY)

    # Save remaining
    if pairs:
        save_pairs(pairs, output_path)
        total_generated += len(pairs)

    print(f"\nDone! Generated {total_generated} pairs total.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
