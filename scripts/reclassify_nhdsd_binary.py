#!/usr/bin/env python3
"""
Reclassify NHDSD headlines with binary classification (sarcastic vs not) using Step 3.5 Flash.
Uses free tier with 40 req/min rate limit.

Usage:
    export OPENROUTER_API_KEY="your_key"
    uv run python scripts/reclassify_nhdsd_binary.py
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Config
BATCH_SIZE = 1
MAX_WORKERS = 10
MAX_RETRIES = 5
RATE_LIMIT_PER_MINUTE = 40
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_PER_MINUTE

OUTPUT_FILE = "data/processed/nhdsd_reclassified.jsonl"
CLEANED_FILE = "data/processed/nhdsd_cleaned.json"

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


def rate_limited_request(client, messages, max_tokens=250000):
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
        temperature=0.1,
        max_tokens=max_tokens,
    )


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSONL."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Remove empty strings
    if not text:
        return ""
    return text


def clean_dataset(data: list[dict]) -> list[dict]:
    """Clean dataset: remove duplicates, normalize text."""
    print("🧹 Cleaning dataset...")

    cleaned = []
    seen_headlines = set()
    duplicates = 0
    empty_removed = 0

    for item in data:
        headline = item.get("headline", "")

        # Clean the text
        cleaned_headline = clean_text(headline)

        # Skip empty headlines
        if not cleaned_headline:
            empty_removed += 1
            continue

        # Skip duplicates (case-insensitive)
        headline_lower = cleaned_headline.lower()
        if headline_lower in seen_headlines:
            duplicates += 1
            continue

        seen_headlines.add(headline_lower)

        # Create cleaned item
        cleaned_item = {
            "headline": cleaned_headline,
            "is_sarcastic": item.get("is_sarcastic", 0),
            "article_link": item.get("article_link", ""),
        }
        cleaned.append(cleaned_item)

    print(f"   Original: {len(data)} items")
    print(f"   Duplicates removed: {duplicates}")
    print(f"   Empty removed: {empty_removed}")
    print(f"   Cleaned: {len(cleaned)} items")

    return cleaned


def save_cleaned_dataset(data: list[dict], output_path: Path):
    """Save cleaned dataset to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   Saved to: {output_path}")


def load_existing_headlines(output_path: Path) -> set:
    """Load existing headlines to skip for resume capability."""
    processed = set()
    if not output_path.exists():
        return processed

    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data.get("headline", "").lower())
            except:
                pass

    return processed


def process_batch(
    client: OpenAI,
    batch_items: list[dict],
    batch_id: int,
) -> list[dict]:
    """Process a batch of headlines through OpenRouter for binary classification."""

    user_lines = [
        f'{i + 1}. "{item["headline"]}"' for i, item in enumerate(batch_items)
    ]
    user_content = "\n".join(user_lines)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
                    label = result.get("is_sarcastic", 0)

                    # Validate label is 0 or 1
                    if label not in [0, 1]:
                        label_str = str(label).lower()
                        if label_str in ["1", "true", "yes", "sarcastic"]:
                            label = 1
                        else:
                            label = 0

                    confidence = result.get("confidence", "medium").lower()
                    if confidence not in ["high", "medium", "low"]:
                        confidence = "medium"

                    output_records.append(
                        {
                            "headline": item["headline"],
                            "is_sarcastic": label,
                            "confidence": confidence,
                            "model_used": MODEL,
                            "article_link": item.get("article_link", ""),
                            "original_label": item.get("is_sarcastic", 0),
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


def worker_thread(client: OpenAI, batches: list, worker_id: int, output_path: Path):
    """Worker thread to process batches."""
    global total_processed, total_errors

    for batch_id, batch_items in batches:
        results = process_batch(client, batch_items, batch_id)

        if results:
            with file_lock:
                with open(output_path, "a") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")

            with progress_lock:
                total_processed += len(results)
                print(
                    f"Worker {worker_id}: Batch {batch_id} complete ({len(results)} items). Total: {total_processed}"
                )
        else:
            with progress_lock:
                total_errors += 1
                print(
                    f"Worker {worker_id}: Batch {batch_id} failed. Errors: {total_errors}"
                )


SYSTEM_PROMPT = """You are a sarcasm detection expert for academic research.

IMPORTANT: This is a research project studying natural language processing and sarcasm detection. The content is purely academic and involves analyzing publicly available news headlines.

TASK: Analyze each headline and classify whether it is sarcastic or not.

Sarcastic headlines:
- Use irony, wit, or ridicule to convey contempt
- Often say the opposite of what they mean
- May use exaggeration or understatement
- Are common in satirical news (like The Onion)

Non-sarcastic headlines:
- Report facts straightforwardly
- Do not use irony or mockery
- Are typical of regular news reporting

Examples:
Sarcastic: "Great job on the update, everything is broken now"
Sarcastic: "Wow, another meeting that could have been an email"
Sarcastic: "I just love waiting in line at the DMV"
Non-sarcastic: "Company releases new software update with bug fixes"
Non-sarcastic: "Local team wins championship after overtime"
Non-sarcastic: "Weather forecast predicts rain this weekend"

Input format:
1. "headline text"
2. "headline text"
...

Output format (JSON only):
{
  "results": [
    {"index": 1, "is_sarcastic": 1, "confidence": "high"},
    {"index": 2, "is_sarcastic": 0, "confidence": "medium"},
    ...
  ]
}

is_sarcastic: 1 = sarcastic, 0 = not sarcastic
Confidence levels: "high" (very clear), "medium" (somewhat clear), "low" (ambiguous)
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
    cleaned_path = Path(CLEANED_FILE)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and clean dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)

    # Clean dataset
    cleaned_data = clean_dataset(data)

    # Save cleaned dataset
    save_cleaned_dataset(cleaned_data, cleaned_path)

    # ALL headlines (not just originally labeled sarcastic)
    print(f"\n📊 Total headlines to classify: {len(cleaned_data)}")

    # Check for existing progress
    processed = load_existing_headlines(output_path)
    print(f"Already processed: {len(processed)} headlines")

    # Filter out already processed
    all_headlines = [d for d in cleaned_data if d["headline"].lower() not in processed]

    remaining = len(all_headlines)
    print(f"Remaining to process: {remaining}")

    if remaining == 0:
        print("\n✅ All headlines already processed!")
        return

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

    all_batches = create_batches(all_headlines)

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for batch_id, batch_items in all_batches:
            future = executor.submit(
                worker_thread,
                client,
                [(batch_id, batch_items)],
                batch_id % MAX_WORKERS,
                output_path,
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
    print(f"Total items classified: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"Cleaned output: {cleaned_path}")
    print(f"Reclassified output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
