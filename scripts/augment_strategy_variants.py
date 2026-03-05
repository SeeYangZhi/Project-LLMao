#!/usr/bin/env python3
"""
Generate 5 missing strategy variants for each non_to_sarcastic pair.

Usage:
    export OPENROUTER_API_KEY="your_key"
    uv run python scripts/augment_strategy_variants.py
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
BATCH_SIZE = 1  # Number of source headlines per batch
MAX_WORKERS = 5
MAX_RETRIES = 5
RATE_LIMIT_PER_MINUTE = 40
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_PER_MINUTE

INPUT_FILE = "data/processed/sarcasm_pairs_step35_clean.jsonl"
OUTPUT_FILE = "data/processed/sarcasm_pairs_strategy_augmented.jsonl"
STATS_FILE = "data/processed/strategy_augmentation_stats.json"

# Model
MODEL = "stepfun/step-3.5-flash:free"

# All 6 strategies
ALL_STRATEGIES = [
    "sarcasm",
    "irony",
    "satire",
    "understatement",
    "overstatement",
    "rhetorical_question",
]

# Thread-safe locks
file_lock = threading.Lock()
progress_lock = threading.Lock()

# Shared counters
total_processed = 0
total_variants_generated = 0
total_errors = 0
last_request_time = 0


def get_openrouter_client():
    """Create OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def rate_limited_request(client, messages, max_tokens=200000):
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
        temperature=0.8,
        max_tokens=max_tokens,
    )


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_existing_sources(output_path: Path) -> set:
    """Load already processed source headlines to skip."""
    processed = set()
    if not output_path.exists():
        return processed

    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Use original_headline + existing_strategy as unique key
                key = f"{data.get('original_headline', '')}_{data.get('existing_strategy', '')}"
                processed.add(key.lower())
            except:
                pass

    return processed


def build_prompt(batch_items: list[dict]) -> str:
    """Build prompt for generating strategy variants."""
    lines = []
    for i, item in enumerate(batch_items, 1):
        original = item.get("original_headline", "")
        existing = item.get("generated_headline", "")
        existing_strategy = item.get("strategy", "satire")
        missing = [s for s in ALL_STRATEGIES if s != existing_strategy]

        lines.append(f'{i}. Original: "{original}"')
        lines.append(f'   Existing ({existing_strategy}): "{existing}"')
        lines.append(f"   Generate 5 variants with: {', '.join(missing)}")
        lines.append("")

    user_content = "\n".join(lines)

    return SYSTEM_PROMPT_TEMPLATE.format(batch_content=user_content)


def process_batch(
    client: OpenAI,
    batch_items: list[dict],
    batch_id: int,
) -> list[dict]:
    """Process a batch of headlines to generate strategy variants."""

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_prompt(batch_items)},
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
                    variants = result.get("variants", [])

                    for variant in variants:
                        strategy = variant.get("strategy", "")
                        headline = variant.get("headline", "")

                        if strategy and headline and strategy in ALL_STRATEGIES:
                            output_records.append(
                                {
                                    "original_headline": item.get(
                                        "original_headline", ""
                                    ),
                                    "non_sarcastic_source": item.get(
                                        "original_headline", ""
                                    ),
                                    "existing_sarcastic": item.get(
                                        "generated_headline", ""
                                    ),
                                    "existing_strategy": item.get("strategy", ""),
                                    "generated_headline": headline,
                                    "strategy": strategy,
                                    "type": "strategy_variant",
                                    "model_used": MODEL,
                                    "article_link": item.get("article_link", ""),
                                    "batch_id": batch_id,
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


def build_system_prompt() -> str:
    """Build the system prompt for strategy variant generation."""
    return """You are a sarcasm style expert for academic research.

TASK: For each headline pair, generate 5 NEW sarcastic variants using different strategies.

The 6 strategies:
1. sarcasm - Contradicts reality with critical tone. Example: "Great job, everything is broken now"
2. irony - Contradicts reality without obvious blame. Example: "I love waiting in line at the DMV"
3. satire - Mock-praise revealing absurdity. Example: "Wow, another meeting that could have been an email"
4. understatement - Severe minimization. Example: "It's just a minor setback" (for disaster)
5. overstatement - Obvious exaggeration. Example: "I've told you a million times!"
6. rhetorical_question - Question implying "obviously no". Example: "Isn't it wonderful to be ignored?"

RULES:
- Generate exactly 5 variants per item
- Each variant must use a DIFFERENT strategy
- Do NOT reuse the existing strategy
- Output valid JSON only
- Be creative and ensure each variant genuinely uses its assigned strategy

Output format:
{
  "results": [
    {
      "index": 1,
      "variants": [
        {"strategy": "sarcasm", "headline": "..."},
        {"strategy": "irony", "headline": "..."},
        {"strategy": "understatement", "headline": "..."},
        {"strategy": "overstatement", "headline": "..."},
        {"strategy": "rhetorical_question", "headline": "..."}
      ]
    },
    ...
  ]
}"""


SYSTEM_PROMPT_TEMPLATE = """{batch_content}"""


def worker_thread(client: OpenAI, batches: list, worker_id: int, output_path: Path):
    """Worker thread to process batches."""
    global total_processed, total_variants_generated, total_errors

    for batch_id, batch_items in batches:
        results = process_batch(client, batch_items, batch_id)

        if results:
            with file_lock:
                with open(output_path, "a") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")

            with progress_lock:
                total_processed += len(batch_items)
                total_variants_generated += len(results)
                print(
                    f"Worker {worker_id}: Batch {batch_id} complete ({len(results)} variants from {len(batch_items)} sources). "
                    f"Total: {total_variants_generated} variants"
                )
        else:
            with progress_lock:
                total_errors += 1
                print(
                    f"Worker {worker_id}: Batch {batch_id} failed. Errors: {total_errors}"
                )


def generate_stats(output_path: Path, input_count: int):
    """Generate statistics about the augmentation."""
    from collections import Counter

    strategy_counts = Counter()
    existing_strategy_counts = Counter()

    with open(output_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                strategy_counts[item.get("strategy", "unknown")] += 1
                existing_strategy_counts[item.get("existing_strategy", "unknown")] += 1
            except:
                pass

    stats = {
        "metadata": {
            "source_pairs": input_count,
            "variants_generated": sum(strategy_counts.values()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "strategy_distribution": dict(strategy_counts),
        "by_existing_strategy": dict(existing_strategy_counts),
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Statistics saved to: {STATS_FILE}")
    print(f"   Total variants: {stats['metadata']['variants_generated']}")
    print("   Strategy distribution:")
    for strategy, count in strategy_counts.most_common():
        print(f"     {strategy}: {count}")


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not set!")
        print("   export OPENROUTER_API_KEY='your_key'")
        return

    client = get_openrouter_client()

    # Paths
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all pairs
    print(f"Loading pairs from {input_path}...")
    all_pairs = load_jsonl(input_path)
    print(f"Total pairs: {len(all_pairs)}")

    # Filter to non_to_sarcastic only
    target_pairs = [p for p in all_pairs if p.get("type") == "non_to_sarcastic"]
    print(f"Non-to-sarcastic pairs: {len(target_pairs)}")

    # Check for existing progress
    processed = load_existing_sources(output_path)
    print(f"Already processed: {len(processed)} sources")

    # Filter out already processed
    remaining = []
    for p in target_pairs:
        key = f"{p.get('original_headline', '')}_{p.get('strategy', '')}".lower()
        if key not in processed:
            remaining.append(p)

    remaining_count = len(remaining)
    print(f"Remaining to process: {remaining_count}")

    if remaining_count == 0:
        print("\n✅ All sources already processed!")
        generate_stats(output_path, len(target_pairs))
        return

    # Estimate time
    total_batches = (remaining_count + BATCH_SIZE - 1) // BATCH_SIZE
    est_minutes = (total_batches * RATE_LIMIT_DELAY) / 60
    expected_variants = remaining_count * 5

    print("\n📊 Processing Plan:")
    print(f"   Source headlines: {remaining_count}")
    print(f"   Expected variants: ~{expected_variants}")
    print(f"   Total batches: {total_batches}")
    print(f"   Batch size: {BATCH_SIZE} sources")
    print(f"   Rate limit: {RATE_LIMIT_PER_MINUTE} req/min")
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

    all_batches = create_batches(remaining)

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
    print(f"Source headlines processed: {total_processed}")
    print(f"Total variants generated: {total_variants_generated}")
    print(f"Errors: {total_errors}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    # Generate stats
    generate_stats(output_path, len(target_pairs))


if __name__ == "__main__":
    main()
