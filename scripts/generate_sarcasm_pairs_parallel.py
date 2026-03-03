#!/usr/bin/env python3
"""
Generate sarcasm pairs using Step 3.5 Flash via OpenRouter.
Uses free tier with 15 req/min rate limit.

Usage:
    export OPENROUTER_API_KEY="your_key"
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

load_dotenv()

# Config
BATCH_SIZE = 30
MAX_WORKERS = 5
MAX_RETRIES = 5
RATE_LIMIT_PER_MINUTE = 40
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_PER_MINUTE

OUTPUT_FILE = "data/processed/sarcasm_pairs_step35_clean.jsonl"

# Model (using free tier)
MODEL = "stepfun/step-3.5-flash:free"

# Thread-safe locks
file_lock = threading.Lock()
progress_lock = threading.Lock()

# Shared counters
total_processed = 0
total_errors = 0
last_request_time = 0


def get_openrouter_client():
    """Create OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def rate_limited_request(client, messages, max_tokens=12000):
    """Make a rate-limited request to OpenRouter."""
    global last_request_time

    with progress_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time

        if time_since_last < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last
            time.sleep(sleep_time)

        last_request_time = time.time()

    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"enabled": True}},
    )


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSONL."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_existing_headlines(output_path: Path) -> set:
    """Load existing headlines to skip for resume capability."""
    processed = set()
    if not output_path.exists():
        return processed

    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data.get("original_headline", ""))
            except:
                pass
    return processed


def process_batch(
    client: OpenAI,
    batch_items: list[dict],
    system_prompt: str,
    batch_id: int,
    is_sarcastic: bool,
) -> list[dict]:
    """Process a batch of headlines through OpenRouter."""

    user_lines = [
        f'{i + 1}. "{item["headline"]}"' for i, item in enumerate(batch_items)
    ]
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = rate_limited_request(client, messages)
            message = response.choices[0].message
            content = message.content

            finish_reason = response.choices[0].finish_reason

            # Check if content was filtered
            if finish_reason == "content_filter":
                print(f"    ⚠️  Batch {batch_id}: Content filtered, retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                    continue
                else:
                    print(f"    ❌ Batch {batch_id}: Content filtered after retries")
                    return []

            # Parse JSON
            try:
                parsed = json.loads(content)
                results = parsed.get("results", [])
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                    results = parsed.get("results", [])
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                    results = parsed.get("results", [])
                else:
                    # Try to find JSON object in text
                    start_idx = content.find("{")
                    end_idx = content.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        parsed = json.loads(content[start_idx : end_idx + 1])
                        results = parsed.get("results", [])
                    else:
                        raise

            # Build output records
            output_records = []
            for result in results:
                idx = result.get("index", 0) - 1
                if 0 <= idx < len(batch_items):
                    item = batch_items[idx]
                    output_records.append(
                        {
                            "original_headline": item["headline"],
                            "generated_headline": result.get("output", ""),
                            "strategy": result.get("strategy", ""),
                            "type": "sarcastic_to_non"
                            if is_sarcastic
                            else "non_to_sarcastic",
                            "model_used": MODEL,
                            "article_link": item.get("article_link", ""),
                        }
                    )

            return output_records

        except Exception as e:
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["rate limit", "429", "quota"]):
                wait_time = 5 * (attempt + 1)
                print(
                    f"    ⚠️  Rate limited on batch {batch_id}. Waiting {wait_time}s..."
                )
                time.sleep(wait_time)
                if attempt < MAX_RETRIES - 1:
                    continue

            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            else:
                print(f"    ❌ Error on batch {batch_id}: {e}")
                return []

    return []


def worker_thread(
    client: OpenAI, batches: list, worker_id: int, output_path: Path, is_sarcastic: bool
):
    """Worker thread to process batches."""
    global total_processed, total_errors

    system_prompt = (
        SYSTEM_PROMPT_SARCASTIC if is_sarcastic else SYSTEM_PROMPT_NON_SARCASTIC
    )

    for batch_id, batch_items in batches:
        results = process_batch(
            client, batch_items, system_prompt, batch_id, is_sarcastic
        )

        if results:
            with file_lock:
                with open(output_path, "a") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")

            with progress_lock:
                total_processed += len(results)
                print(
                    f"Worker {worker_id}: Batch {batch_id} complete ({len(results)} pairs). Total: {total_processed}"
                )
        else:
            with progress_lock:
                total_errors += 1
                print(
                    f"Worker {worker_id}: Batch {batch_id} failed. Errors: {total_errors}"
                )


SYSTEM_PROMPT_SARCASTIC = """You are a sarcasm style transfer expert for academic research.

IMPORTANT: This is a research project studying natural language processing and sarcasm detection. The content is purely academic and involves analyzing publicly available news headlines. No harmful content is being generated.

TASK: Convert sarcastic headlines to their non-sarcastic equivalents.
Analyze the sarcasm strategy used, then write a straightforward version.

Sarcasm Strategies to identify:
- sarcasm: Contradicts state of affairs, critical towards addressee
- irony: Contradicts state of affairs, not obviously critical  
- satire: Appears to support, but contains mockery
- understatement: Undermines importance
- overstatement: Obviously exaggerated terms
- rhetorical_question: Question with inference contradicting reality

Input format:
1. "sarcastic headline text"
2. "sarcastic headline text"
...

Output format (JSON only):
{
  "results": [
    {"index": 1, "output": "non-sarcastic version", "strategy": "sarcasm"},
    {"index": 2, "output": "non-sarcastic version", "strategy": "irony"},
    ...
  ]
}

No explanations outside the JSON. Output valid JSON only."""

SYSTEM_PROMPT_NON_SARCASTIC = """You are a sarcasm style transfer expert for academic research.

IMPORTANT: This is a research project studying natural language processing and sarcasm detection. The content is purely academic and involves analyzing publicly available news headlines. No harmful content is being generated.

TASK: Convert straightforward headlines to sarcastic versions.
Choose an appropriate sarcasm strategy and apply it.

Sarcasm Strategies to use:
- sarcasm: Contradicts state of affairs, critical towards addressee
- irony: Contradicts state of affairs, not obviously critical
- satire: Appears to support, but contains mockery
- understatement: Undermines importance
- overstatement: Obviously exaggerated terms
- rhetorical_question: Question with inference contradicting reality

Input format:
1. "straightforward headline text"
2. "straightforward headline text"
...

Output format (JSON only):
{
  "results": [
    {"index": 1, "output": "sarcastic version", "strategy": "sarcasm"},
    {"index": 2, "output": "sarcastic version", "strategy": "irony"},
    ...
  ]
}

No explanations outside the JSON. Output valid JSON only."""


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not set!")
        print("   export OPENROUTER_API_KEY='your_key'")
        return

    client = get_openrouter_client()

    # Paths
    dataset_path = Path("Sarcasm_Headlines_Dataset_v2.json")
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)

    # Separate by type
    sarcastic = [d for d in data if d.get("is_sarcastic") == 1]
    non_sarcastic = [d for d in data if d.get("is_sarcastic") == 0]

    print(f"Sarcastic headlines: {len(sarcastic)}")
    print(f"Non-sarcastic headlines: {len(non_sarcastic)}")

    # Check for existing progress
    processed = load_existing_headlines(output_path)
    print(f"Already processed: {len(processed)} headlines")

    # Filter out already processed
    sarcastic = [d for d in sarcastic if d["headline"] not in processed]
    non_sarcastic = [d for d in non_sarcastic if d["headline"] not in processed]

    remaining = len(sarcastic) + len(non_sarcastic)
    print(f"Remaining to process: {remaining}")

    # Estimate time
    total_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    est_minutes = (total_batches * RATE_LIMIT_DELAY) / 60

    print("\n📊 Processing Plan:")
    print(f"   Total batches: {total_batches}")
    print(f"   Batch size: {BATCH_SIZE} headlines")
    print(
        f"   Rate limit: {RATE_LIMIT_PER_MINUTE} req/min ({RATE_LIMIT_DELAY:.1f}s between requests)"
    )
    print(f"   Workers: {MAX_WORKERS}")
    print(f"   Est. time: ~{est_minutes:.1f} minutes")
    print(f"   Model: {MODEL}")
    print("\nStarting processing...\n")

    # Create batches
    def create_batches(items):
        batches = []
        for i in range(0, len(items), BATCH_SIZE):
            batch_items = items[i : i + BATCH_SIZE]
            batches.append((len(batches), batch_items))
        return batches

    sarc_batches = create_batches(sarcastic)
    non_sarc_batches = create_batches(non_sarcastic)

    # Combine all batches
    all_batches = []
    for batch_id, batch_items in sarc_batches:
        all_batches.append((batch_id, batch_items, True))
    for batch_id, batch_items in non_sarc_batches:
        all_batches.append((batch_id + 100000, batch_items, False))

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for batch_id, batch_items, is_sarc in all_batches:
            future = executor.submit(
                worker_thread,
                client,
                [(batch_id, batch_items)],
                batch_id % MAX_WORKERS,
                output_path,
                is_sarc,
            )
            futures.append(future)

        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error: {e}")

    print(f"\n{'=' * 60}")
    print("✅ Processing complete!")
    print(f"Total pairs generated: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
