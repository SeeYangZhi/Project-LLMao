#!/usr/bin/env python3
"""
Generate ONLY the missing strategy variants for 508 incomplete sources.

Unlike augment_strategy_variants.py which generates all 5 missing strategies,
this script reads incomplete_strategy_sources.jsonl and generates only the
specific 1-2 strategies that are missing per source.

Usage:
    export OPENROUTER_API_KEY="your_key"
    uv run python scripts/augment_incomplete_sources.py
"""

import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Config
BATCH_SIZE = 5  # Headlines per API call (higher since fewer variants per item)
MAX_WORKERS = 5
MAX_RETRIES = 5
RATE_LIMIT_PER_MINUTE = 40
RATE_LIMIT_DELAY = 60.0 / RATE_LIMIT_PER_MINUTE

INPUT_FILE = "data/processed/incomplete_strategy_sources.jsonl"
OUTPUT_FILE = "data/processed/sarcasm_pairs_strategy_augmented_patch.jsonl"
MAIN_AUGMENTED_FILE = "data/processed/sarcasm_pairs_strategy_augmented.jsonl"

# Model (same as original augmentation)
MODEL = "stepfun/step-3.5-flash:free"

ALL_STRATEGIES = [
    "sarcasm",
    "irony",
    "satire",
    "understatement",
    "overstatement",
    "rhetorical_question",
]

# Strategy definitions for prompting
STRATEGY_DEFINITIONS = {
    "sarcasm": "Contradicts reality with critical tone. Says the opposite of what's meant, with a biting edge. Example: 'Great job, everything is broken now'",
    "irony": "Contradicts reality without obvious blame. Says X when clearly not-X, no direct criticism. Example: 'I love waiting in line at the DMV'",
    "satire": "Mock-praise revealing absurdity. Appears supportive but contains mockery. Example: 'Wow, another meeting that could have been an email'",
    "understatement": "Severe minimization. Dramatically undermines importance of something significant. Example: 'It's just a minor setback' (for a disaster)",
    "overstatement": "Obvious exaggeration. Maximizes severity unrealistically. Example: 'I've told you a million times!'",
    "rhetorical_question": "Question implying 'obviously no/yes'. The implied answer contradicts reality. Example: 'Isn't it wonderful to be ignored?'",
}

# Thread-safe locks
file_lock = threading.Lock()
progress_lock = threading.Lock()
rate_lock = threading.Lock()

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

    with rate_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)
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


def load_already_patched(output_path: Path) -> set:
    """Load already patched source+strategy combos to enable resume."""
    patched = set()
    if not output_path.exists():
        return patched

    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                key = f"{data.get('original_headline', '')}||{data.get('strategy', '')}"
                patched.add(key.lower())
            except Exception:
                pass
    return patched


def build_system_prompt() -> str:
    """Build the system prompt for targeted strategy generation."""
    return """You are a sarcasm style expert for academic research.

TASK: For each headline, generate sarcastic variants using ONLY the specified strategies.

The 6 strategies:
1. sarcasm - Contradicts reality with critical tone. Example: "Great job, everything is broken now"
2. irony - Contradicts reality without obvious blame. Example: "I love waiting in line at the DMV"
3. satire - Mock-praise revealing absurdity. Example: "Wow, another meeting that could have been an email"
4. understatement - Severe minimization. Example: "It's just a minor setback" (for disaster)
5. overstatement - Obvious exaggeration. Example: "I've told you a million times!"
6. rhetorical_question - Question implying "obviously no". Example: "Isn't it wonderful to be ignored?"

RULES:
- Generate ONLY the requested strategies (not all 6)
- Each variant must genuinely use its assigned strategy
- Be creative and varied
- Output valid JSON only

Output format:
{
  "results": [
    {
      "index": 1,
      "variants": [
        {"strategy": "overstatement", "headline": "..."}
      ]
    }
  ]
}"""


def build_user_prompt(batch_items: list[dict]) -> str:
    """Build user prompt for a batch of incomplete sources."""
    lines = []
    for i, item in enumerate(batch_items, 1):
        original = item.get("original_headline", "")
        existing_sarcastic = item.get("generated_headline", "")
        existing_strategy = item.get("existing_strategy", "")
        missing = item.get("missing_strategies", [])

        strategy_descriptions = []
        for s in missing:
            desc = STRATEGY_DEFINITIONS.get(s, s)
            strategy_descriptions.append(f"  - {s}: {desc}")

        lines.append(f'{i}. Original (non-sarcastic): "{original}"')
        lines.append(
            f'   Existing sarcastic ({existing_strategy}): "{existing_sarcastic}"'
        )
        lines.append(
            f"   Generate variants for these {len(missing)} missing strategies:"
        )
        lines.extend(strategy_descriptions)
        lines.append("")

    return "\n".join(lines)


def parse_response(content: str) -> dict:
    """Parse JSON from LLM response with fallbacks."""
    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0].strip()
        return json.loads(json_str)
    if "```" in content:
        json_str = content.split("```")[1].split("```")[0].strip()
        return json.loads(json_str)

    # Try to find JSON object
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1:
        return json.loads(content[start_idx : end_idx + 1])

    raise json.JSONDecodeError("No JSON found", content, 0)


def process_batch(
    client: OpenAI,
    batch_items: list[dict],
    batch_id: int,
) -> list[dict]:
    """Process a batch to generate missing strategy variants."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(batch_items)},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = rate_limited_request(client, messages)
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "content_filter":
                print(f"    ⚠️  Batch {batch_id}: Content filtered, retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                    continue
                else:
                    print(f"    ❌ Batch {batch_id}: Content filtered after retries")
                    return []

            if not content:
                print(f"    ⚠️  Batch {batch_id}: Empty response, retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                else:
                    return []

            parsed = parse_response(content)
            results = parsed.get("results", [])

            output_records = []
            for result in results:
                idx = result.get("index", 0) - 1
                if 0 <= idx < len(batch_items):
                    item = batch_items[idx]
                    variants = result.get("variants", [])
                    expected_missing = set(item.get("missing_strategies", []))

                    for variant in variants:
                        strategy = variant.get("strategy", "")
                        headline = variant.get("headline", "")

                        # Only accept variants for actually missing strategies
                        if (
                            strategy
                            and headline
                            and strategy in ALL_STRATEGIES
                            and strategy in expected_missing
                        ):
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
                                    "existing_strategy": item.get(
                                        "existing_strategy", ""
                                    ),
                                    "generated_headline": headline,
                                    "strategy": strategy,
                                    "type": "strategy_variant",
                                    "variant_type": "patch",
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


def worker_fn(client: OpenAI, batches: list, worker_id: int, output_path: Path):
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
                    f"Worker {worker_id}: Batch {batch_id} → {len(results)} variants "
                    f"from {len(batch_items)} sources. Total: {total_variants_generated}"
                )
        else:
            with progress_lock:
                total_errors += 1
                print(
                    f"Worker {worker_id}: Batch {batch_id} failed. Errors: {total_errors}"
                )


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not set!")
        print("   export OPENROUTER_API_KEY='your_key'")
        return

    client = get_openrouter_client()

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load incomplete sources
    print(f"Loading incomplete sources from {input_path}...")
    sources = load_jsonl(INPUT_FILE)
    print(f"  Total incomplete sources: {len(sources)}")

    # Count expected variants
    total_missing = sum(len(s.get("missing_strategies", [])) for s in sources)
    print(f"  Total missing variants to generate: {total_missing}")

    # Missing breakdown
    missing_counts = Counter()
    for s in sources:
        for strategy in s.get("missing_strategies", []):
            missing_counts[strategy] += 1
    print("  Missing by strategy:")
    for strategy, count in missing_counts.most_common():
        print(f"    {strategy}: {count}")

    # Check resume state
    already_patched = load_already_patched(output_path)
    print(f"\n  Already patched: {len(already_patched)} strategy variants")

    # Filter to remaining work
    remaining = []
    for s in sources:
        # Check which strategies for this source are still missing
        still_missing = []
        for strategy in s.get("missing_strategies", []):
            key = f"{s.get('original_headline', '')}||{strategy}".lower()
            if key not in already_patched:
                still_missing.append(strategy)

        if still_missing:
            item = dict(s)
            item["missing_strategies"] = still_missing
            remaining.append(item)

    remaining_variants = sum(len(s.get("missing_strategies", [])) for s in remaining)
    print(f"  Remaining sources to process: {len(remaining)}")
    print(f"  Remaining variants to generate: {remaining_variants}")

    if not remaining:
        print("\n✅ All incomplete sources already patched!")
        return

    # Create batches
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    est_minutes = (total_batches * RATE_LIMIT_DELAY) / 60

    print(f"\n📊 Processing Plan:")
    print(f"   Sources: {len(remaining)}")
    print(f"   Expected variants: ~{remaining_variants}")
    print(f"   Total batches: {total_batches}")
    print(f"   Batch size: {BATCH_SIZE} sources")
    print(f"   Rate limit: {RATE_LIMIT_PER_MINUTE} req/min")
    print(f"   Workers: {MAX_WORKERS}")
    print(f"   Est. time: ~{est_minutes:.1f} minutes")
    print(f"   Model: {MODEL}")
    print("\nStarting processing...\n")

    batches = []
    for i in range(0, len(remaining), BATCH_SIZE):
        batch_items = remaining[i : i + BATCH_SIZE]
        batches.append((len(batches), batch_items))

    # Distribute batches across workers
    worker_batches = [[] for _ in range(MAX_WORKERS)]
    for i, batch in enumerate(batches):
        worker_batches[i % MAX_WORKERS].append(batch)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for worker_id, worker_batch in enumerate(worker_batches):
            if worker_batch:
                future = executor.submit(
                    worker_fn, client, worker_batch, worker_id, output_path
                )
                futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error: {e}")

    print(f"\n{'=' * 60}")
    print("✅ Patch processing complete!")
    print(f"Sources processed: {total_processed}")
    print(f"Variants generated: {total_variants_generated}")
    print(f"Errors: {total_errors}")
    print(f"Patch output: {output_path}")
    print(f"{'=' * 60}")

    # Merge patch into main augmented file (idempotent: dedup by headline+strategy)
    print(f"\nMerging patch into {MAIN_AUGMENTED_FILE}...")
    patch_records = load_jsonl(str(output_path))

    existing_keys = set()
    if Path(MAIN_AUGMENTED_FILE).exists():
        with open(MAIN_AUGMENTED_FILE, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    key = f"{item.get('original_headline', '')}||{item.get('strategy', '')}"
                    existing_keys.add(key.lower())
                except Exception:
                    pass

    new_records = []
    for record in patch_records:
        key = f"{record.get('original_headline', '')}||{record.get('strategy', '')}"
        if key.lower() not in existing_keys:
            new_records.append(record)

    with open(MAIN_AUGMENTED_FILE, "a") as f:
        for record in new_records:
            f.write(json.dumps(record) + "\n")

    print(f"  Appended {len(new_records)} new records to {MAIN_AUGMENTED_FILE} (skipped {len(patch_records) - len(new_records)} duplicates)")

    # Verify final counts
    print("\nVerifying augmented file...")
    all_augmented = load_jsonl(MAIN_AUGMENTED_FILE)
    by_source = defaultdict(set)
    for item in all_augmented:
        headline = item.get("original_headline", "")
        strategy = item.get("strategy", "")
        by_source[headline].add(strategy)

    strategy_counts = Counter()
    for item in all_augmented:
        strategy_counts[item.get("strategy", "unknown")] += 1

    complete_5 = sum(1 for strategies in by_source.values() if len(strategies) == 5)
    incomplete = sum(1 for strategies in by_source.values() if len(strategies) < 5)

    print(f"  Total augmented records: {len(all_augmented)}")
    print(f"  Unique sources: {len(by_source)}")
    print(f"  Sources with 5 augmented strategies: {complete_5}")
    print(f"  Sources still incomplete: {incomplete}")
    print(f"  Strategy distribution:")
    for strategy, count in strategy_counts.most_common():
        print(f"    {strategy}: {count}")

    if incomplete > 0:
        print(f"\n⚠️  {incomplete} sources still incomplete after patching.")
        print("  These may need manual review or another run.")
    else:
        print("\n✅ All sources now have complete strategy coverage!")

    print(f"\nNext step: uv run python scripts/merge_augmented_variants.py")


if __name__ == "__main__":
    main()
