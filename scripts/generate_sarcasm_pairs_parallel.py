#!/usr/bin/env python3
"""
Generate sarcasm pairs using MiniMax M2.5 via OpenCode Zen API.
Parallel version with conservative rate limiting.

Usage:
    uv run python scripts/generate_sarcasm_pairs_parallel.py
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env
load_dotenv()

# Collect all available API keys
api_keys = []
for i in range(0, 100):
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

# Config - Conservative to avoid rate limits
BATCH_SIZE = 50
API_DELAY = 2.0  # 2 second delay between requests per worker
MAX_WORKERS_PER_KEY = 1  # 1 worker per key to avoid rate limits
MAX_RETRIES = 3
RETRY_DELAY = 10.0  # Wait 10 seconds before retry

# Thread-safe lock for file writing
file_lock = threading.Lock()
progress_lock = threading.Lock()

# Shared counters
total_generated = 0

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


def load_existing_headlines(output_path: Path) -> set:
    """Load existing headlines to skip for resume capability."""
    existing = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    pair = json.loads(line)
                    existing.add(pair.get("sarcastic", ""))
                    existing.add(pair.get("non_sarcastic", ""))
                except json.JSONDecodeError:
                    pass
    return existing


def save_batch(pairs: list[dict], output_path: Path):
    """Thread-safe batch save to JSONL."""
    with file_lock:
        with open(output_path, "a") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")


def generate_single(item: dict, direction: str, api_key: str) -> dict:
    """Generate a single pair using specific API key with retries."""
    headline = item["headline"]

    if direction == "sarcastic_to_non":
        user_content = f'Input: Sarcastic: "{headline}"\nOutput:'
    else:
        user_content = f'Input: Non-sarcastic: "{headline}"\nOutput:'

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    client = OpenAI(
        api_key=api_key,
        base_url="https://opencode.ai/zen/v1",
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="minimax-m2.5-free",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            if not result_text:
                return None

            result = json.loads(result_text)

            if direction == "sarcastic_to_non":
                return {
                    "sarcastic": headline,
                    "non_sarcastic": result.get("non_sarcastic", ""),
                    "strategy_detected": result.get("strategy_detected", ""),
                    "direction": direction,
                }
            else:
                return {
                    "sarcastic": result.get("sarcastic", ""),
                    "non_sarcastic": headline,
                    "strategy_applied": result.get("strategy_applied", ""),
                    "direction": direction,
                }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(
                    f"  Retry {attempt + 1}/{MAX_RETRIES} for '{headline[:50]}...': {e}"
                )
                time.sleep(RETRY_DELAY)
            else:
                print(
                    f"  Failed after {MAX_RETRIES} attempts: '{headline[:50]}...': {e}"
                )
                return None
        finally:
            time.sleep(API_DELAY)


def process_batch(
    items: list[dict], direction: str, api_key: str, output_path: Path, desc: str
):
    """Process a batch of items with specific API key."""
    global total_generated

    pairs = []
    for item in items:
        result = generate_single(item, direction, api_key)
        if result:
            pairs.append(result)

    if pairs:
        save_batch(pairs, output_path)
        with progress_lock:
            total_generated += len(pairs)
        print(f"  [{desc}] Saved {len(pairs)} pairs (total: {total_generated})")


def main():
    # Paths
    dataset_path = Path("Sarcasm_Headlines_Dataset_v2.json")
    output_path = Path("data/processed/sarcasm_pairs_minimax.jsonl")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)

    # Separate and filter out existing
    existing = load_existing_headlines(output_path)
    sarcastic_headlines = [
        d for d in data if d.get("is_sarcastic") == 1 and d["headline"] not in existing
    ]
    non_sarcastic_headlines = [
        d for d in data if d.get("is_sarcastic") == 0 and d["headline"] not in existing
    ]

    print(f"Total: {len(data)}")
    print(f"Sarcastic (remaining): {len(sarcastic_headlines)}")
    print(f"Non-sarcastic (remaining): {len(non_sarcastic_headlines)}")
    print(
        f"Using {len(api_keys)} API keys × {MAX_WORKERS_PER_KEY} workers = {len(api_keys) * MAX_WORKERS_PER_KEY} concurrent requests"
    )
    print(
        f"Rate limit: {API_DELAY}s delay between requests, {MAX_RETRIES} retries with {RETRY_DELAY}s backoff"
    )

    if not sarcastic_headlines and not non_sarcastic_headlines:
        print("All items already processed!")
        return

    # Process Sarcastic → Non-sarcastic
    if sarcastic_headlines:
        print(
            f"\nGenerating sarcastic → non-sarcastic ({len(sarcastic_headlines)} pairs)..."
        )
        process_parallel(sarcastic_headlines, "sarcastic_to_non", output_path)

    # Process Non-sarcastic → Sarcastic
    if non_sarcastic_headlines:
        print(
            f"\nGenerating non-sarcastic → sarcastic ({len(non_sarcastic_headlines)} pairs)..."
        )
        process_parallel(non_sarcastic_headlines, "non_to_sarcastic", output_path)

    print(f"\nDone! Generated {total_generated} pairs total.")
    print(f"Output saved to: {output_path}")


def process_parallel(items: list[dict], direction: str, output_path: Path):
    """Process items in parallel using ThreadPoolExecutor."""
    global total_generated

    # Split items into chunks for each worker
    total_workers = len(api_keys) * MAX_WORKERS_PER_KEY
    chunk_size = max(1, len(items) // total_workers)
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    print(f"  Split into {len(chunks)} chunks of ~{chunk_size} items each")

    # Assign API keys round-robin to workers
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = []

        for i, chunk in enumerate(chunks):
            api_key = api_keys[i % len(api_keys)]
            worker_id = f"W{i + 1}"
            future = executor.submit(
                process_batch, chunk, direction, api_key, output_path, worker_id
            )
            futures.append(future)

        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Worker error: {e}")


if __name__ == "__main__":
    main()
