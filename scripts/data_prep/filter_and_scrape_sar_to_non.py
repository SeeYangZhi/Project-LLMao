"""Filter sar-to-non pairs: cross-validated sarcastic only + scrape articles to remove headline-only memes.

Pipeline:
1. Load sarcasm_pairs_sarcastic_to_non.jsonl (13,588 records)
2. Filter to only cross-validated sarcastic headlines (agreed + CV confirmed)
3. Normalize TheOnion subdomain URLs → theonion.com
4. Scrape each article to check for body content
5. Filter out headline-only/meme articles (no body text)
6. Output filtered dataset + train/val/test splits

Usage:
    # Step 1: Filter + normalize (fast, no network)
    uv run python scripts/data_prep/filter_and_scrape_sar_to_non.py --step filter

    # Step 2: Scrape articles (slow, needs network, writes cache)
    uv run python scripts/data_prep/filter_and_scrape_sar_to_non.py --step scrape

    # Step 3: Apply scrape results + create splits
    uv run python scripts/data_prep/filter_and_scrape_sar_to_non.py --step split

    # Or run all steps:
    uv run python scripts/data_prep/filter_and_scrape_sar_to_non.py --step all
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERMEDIATE_DIR = PROCESSED_DIR / "intermediate"
OUTPUT_DIR = DATA_DIR / "splits" / "sar_to_non_filtered"

# Input files
PAIRS_FILE = PROCESSED_DIR / "sarcasm_pairs_sarcastic_to_non.jsonl"
RECLASSIFIED_FILE = INTERMEDIATE_DIR / "nhdsd_reclassified.jsonl"
CROSS_VAL_FILE = INTERMEDIATE_DIR / "cross_validation_secondary.jsonl"

# Intermediate outputs
FILTERED_FILE = PROCESSED_DIR / "sarcasm_pairs_sar_to_non_cv_filtered.jsonl"
SCRAPE_CACHE_FILE = INTERMEDIATE_DIR / "article_scrape_cache.jsonl"

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Subdomains to normalize
ONION_SUBDOMAIN_RE = re.compile(
    r"https?://(local|politics|entertainment|sports|ogn|www)\.theonion\.com/"
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def normalize_url(url: str) -> str:
    """Normalize TheOnion subdomain URLs to theonion.com."""
    return ONION_SUBDOMAIN_RE.sub("https://theonion.com/", url)


# ---------------------------------------------------------------------------
# Step 1: Filter to cross-validated sarcastic only + normalize URLs
# ---------------------------------------------------------------------------


def step_filter() -> list[dict]:
    print("=== Step 1: Filter to cross-validated sarcastic ===")

    pairs = load_jsonl(PAIRS_FILE)
    print(f"Loaded {len(pairs)} sar-to-non pairs")

    # Build lookup: headline → reclassified record
    reclassified = {r["headline"]: r for r in load_jsonl(RECLASSIFIED_FILE)}
    print(f"Loaded {len(reclassified)} reclassified records")

    # Build lookup: headline → cross-validation record
    cross_val = {r["headline"]: r for r in load_jsonl(CROSS_VAL_FILE)}
    print(f"Loaded {len(cross_val)} cross-validation records")

    kept = []
    stats = Counter()

    for p in pairs:
        headline = p["original_headline"]
        r = reclassified.get(headline)

        if r is None:
            stats["not_in_reclassified"] += 1
            continue

        if r["is_sarcastic"] == r["original_label"] == 1:
            # Both agreed it's sarcastic
            stats["agreed_sarcastic"] += 1
            kept.append(p)
        elif r["is_sarcastic"] != r["original_label"]:
            # Disagreement — check cross-validation
            cv = cross_val.get(headline)
            if cv and cv["is_sarcastic"] == 1:
                # CV confirmed sarcastic
                stats["cv_confirmed_sarcastic"] += 1
                kept.append(p)
            else:
                stats["cv_confirmed_non_sarcastic"] += 1
        else:
            stats["other"] += 1

    print("\nFilter results:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  TOTAL KEPT: {len(kept)}")

    # Normalize URLs
    url_changes = 0
    for p in kept:
        old_url = p["article_link"]
        new_url = normalize_url(old_url)
        if old_url != new_url:
            p["article_link"] = new_url
            url_changes += 1
    print(f"\nNormalized {url_changes} subdomain URLs → theonion.com")

    save_jsonl(kept, FILTERED_FILE)
    return kept


# ---------------------------------------------------------------------------
# Step 2: Scrape articles to detect headline-only memes
# ---------------------------------------------------------------------------


def scrape_article(url: str, timeout: int = 10) -> dict:
    """Fetch article and detect if it has body content.

    Returns dict with:
        - has_body: bool — True if article has body text paragraphs
        - body_text: str — extracted body text (first 500 chars)
        - status_code: int
        - error: str or None
    """
    import urllib.error
    import urllib.request

    result = {
        "url": url,
        "has_body": False,
        "body_text": "",
        "image_url": "",
        "image_caption": "",
        "status_code": 0,
        "error": None,
    }

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result["status_code"] = resp.status
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        result["status_code"] = e.code
        result["error"] = f"HTTP {e.code}"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

    # Check for wp-block-post-content in article
    article_match = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL)
    if not article_match:
        result["error"] = "no_article_tag"
        return result

    article_html = article_match.group(1)

    # Extract featured image (wp-block-post-featured-image)
    feat_match = re.search(
        r"<figure[^>]*featured-image[^>]*>.*?<img[^>]*src=\"([^\"]+)\"",
        article_html,
        re.DOTALL,
    )
    if feat_match:
        result["image_url"] = feat_match.group(1)
    cap_match = re.search(
        r"<figcaption[^>]*>(.*?)</figcaption>", article_html, re.DOTALL
    )
    if cap_match:
        result["image_caption"] = re.sub(r"<[^>]+>", "", cap_match.group(1)).strip()

    # Check if post-content div exists
    has_content_div = bool(
        re.search(r'class="[^"]*wp-block-post-content[^"]*"', article_html)
    )

    # Extract body paragraphs (>50 chars, not UI elements)
    all_ps = re.findall(r"<p[^>]*>(.*?)</p>", article_html, re.DOTALL)
    body_paragraphs = []
    for p in all_ps:
        clean = re.sub(r"<[^>]+>", "", p).strip()
        if (
            clean
            and len(clean) > 50
            and "Share" not in clean
            and "Published" not in clean
            and "Become A Member" not in clean
        ):
            body_paragraphs.append(clean)

    if body_paragraphs:
        result["has_body"] = True
        result["body_text"] = " ".join(body_paragraphs)
    elif has_content_div:
        # Has content div but no substantial paragraphs — might be image/infographic only
        result["has_body"] = False
        result["body_text"] = ""
    else:
        result["has_body"] = False

    return result


def step_scrape() -> dict[str, dict]:
    print("=== Step 2: Scrape articles ===")

    if not FILTERED_FILE.exists():
        print(f"ERROR: {FILTERED_FILE} not found. Run --step filter first.")
        return {}

    pairs = load_jsonl(FILTERED_FILE)

    # Load existing cache
    cache: dict[str, dict] = {}
    if SCRAPE_CACHE_FILE.exists():
        for r in load_jsonl(SCRAPE_CACHE_FILE):
            cache[r["url"]] = r
        print(f"Loaded {len(cache)} cached scrape results")

    # Get unique URLs to scrape
    # Retry transient errors (timeouts, connection drops) but not permanent 404s
    def should_retry(entry: dict) -> bool:
        err = entry.get("error")
        if not err:
            return False  # success, skip
        return entry.get("status_code") != 404  # retry everything except 404

    urls = list({p["article_link"] for p in pairs})
    to_scrape = [u for u in urls if u not in cache or should_retry(cache[u])]
    print(f"Total unique URLs: {len(urls)}")
    print(f"Already cached: {len(urls) - len(to_scrape)}")
    print(f"To scrape: {len(to_scrape)}")

    if not to_scrape:
        print("Nothing to scrape!")
        return cache

    # Parallel scrape with ThreadPoolExecutor
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    has_body_count = sum(1 for v in cache.values() if v.get("has_body"))
    no_body_count = sum(
        1 for v in cache.values() if not v.get("has_body") and not v.get("error")
    )
    error_count = sum(1 for v in cache.values() if v.get("error"))
    completed = 0
    lock = threading.Lock()
    start_time = time.time()

    WORKERS = 20  # concurrent connections
    print(
        f"Starting parallel scrape ({WORKERS} workers) at {time.strftime('%H:%M:%S')}...",
        flush=True,
    )

    def scrape_and_cache(url: str) -> tuple[str, dict]:
        result = scrape_article(url)
        return url, result

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(scrape_and_cache, url): url for url in to_scrape}

        for future in as_completed(futures):
            url, result = future.result()

            with lock:
                cache[url] = result
                completed += 1

                if result.get("error"):
                    error_count += 1
                    tag = f"ERR:{result['error'][:20]}"
                elif result["has_body"]:
                    has_body_count += 1
                    tag = "BODY"
                else:
                    no_body_count += 1
                    tag = "MEME"

                slug = url.split("/")[-1][:45]
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta_min = (len(to_scrape) - completed) / rate / 60 if rate > 0 else 0
                print(
                    f"  [{completed}/{len(to_scrape)}] [{tag:4s}] {slug}  "
                    f"(body={has_body_count} meme={no_body_count} err={error_count} "
                    f"ETA={eta_min:.0f}m)",
                    flush=True,
                )

                # Save cache every 100 completions
                if completed % 100 == 0:
                    save_jsonl(list(cache.values()), SCRAPE_CACHE_FILE)
                    print(f"  --- cache saved ({len(cache)} entries) ---", flush=True)

    # Final cache save
    save_jsonl(list(cache.values()), SCRAPE_CACHE_FILE)

    # Print stats
    has_body = sum(1 for v in cache.values() if v.get("has_body"))
    no_body = sum(
        1 for v in cache.values() if not v.get("has_body") and not v.get("error")
    )
    errors = sum(1 for v in cache.values() if v.get("error"))
    print("\nScrape results:")
    print(f"  Has body text: {has_body}")
    print(f"  Headline-only (meme): {no_body}")
    print(f"  Errors: {errors}")

    return cache


# ---------------------------------------------------------------------------
# Step 3: Apply scrape filter + create splits
# ---------------------------------------------------------------------------


def stratified_split(
    records: list[dict], ratios: dict[str, float], seed: int
) -> dict[str, list[dict]]:
    """Split records into train/val/test, stratified by strategy."""
    rng = random.Random(seed)

    by_strategy: dict[str, list[dict]] = {}
    for r in records:
        by_strategy.setdefault(r["strategy"], []).append(r)

    splits: dict[str, list[dict]] = {k: [] for k in ratios}

    for strategy, items in sorted(by_strategy.items()):
        rng.shuffle(items)
        n = len(items)
        train_end = int(n * ratios["train"])
        val_end = train_end + int(n * ratios["val"])

        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])

    for items in splits.values():
        rng.shuffle(items)

    return splits


def step_split() -> None:
    print("=== Step 3: Apply scrape filter + create splits ===")

    if not FILTERED_FILE.exists():
        print(f"ERROR: {FILTERED_FILE} not found. Run --step filter first.")
        return

    pairs = load_jsonl(FILTERED_FILE)

    # Load scrape cache
    cache: dict[str, dict] = {}
    if SCRAPE_CACHE_FILE.exists():
        for r in load_jsonl(SCRAPE_CACHE_FILE):
            cache[r["url"]] = r
        print(f"Loaded {len(cache)} scrape results")
    else:
        print("WARNING: No scrape cache found. Skipping body-content filter.")
        print("Run --step scrape first for full filtering.")

    # Enrich records with scraped content and filter out articles with
    # neither body text nor a featured image (no usable content at all).
    if cache:
        final = []
        stats = Counter()
        for p in pairs:
            url = p["article_link"]
            scrape = cache.get(url)
            if scrape is None:
                stats["not_scraped"] += 1
                final.append(p)  # keep if not scraped
            elif scrape.get("error"):
                stats["scrape_error"] += 1
                final.append(p)  # keep if error (benefit of doubt)
            else:
                # Enrich with scraped data
                if scrape.get("body_text"):
                    p["article_body"] = scrape["body_text"]
                if scrape.get("image_url"):
                    p["image_url"] = scrape["image_url"]
                if scrape.get("image_caption"):
                    p["image_caption"] = scrape["image_caption"]

                has_body = scrape["has_body"]
                has_image = bool(scrape.get("image_url"))

                if has_body and has_image:
                    stats["body_and_image"] += 1
                    final.append(p)
                elif has_body:
                    stats["body_only"] += 1
                    final.append(p)
                elif has_image:
                    stats["image_only"] += 1
                    final.append(p)
                else:
                    stats["no_content_filtered"] += 1

        print("\nContent filter:")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
        print(f"  TOTAL KEPT: {len(final)}")
    else:
        final = pairs

    # Save final filtered dataset
    final_file = PROCESSED_DIR / "sarcasm_pairs_sar_to_non_final.jsonl"
    save_jsonl(final, final_file)

    # Create splits
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = stratified_split(final, SPLIT_RATIOS, SEED)

    metadata = {
        "source_file": str(final_file.relative_to(PROJECT_ROOT)),
        "pipeline": "cross-validated + body-content filtered",
        "total_records": len(final),
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "splits": {},
        "strategy_distribution": {},
    }

    for split_name, items in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        save_jsonl(items, out_path)

        strategy_counts = Counter(r["strategy"] for r in items)
        metadata["splits"][split_name] = {
            "count": len(items),
            "strategy_distribution": dict(sorted(strategy_counts.items())),
        }
        print(f"  {split_name}: {len(items)} records")

    overall_dist = Counter(r["strategy"] for r in final)
    metadata["strategy_distribution"] = dict(sorted(overall_dist.items()))

    meta_path = OUTPUT_DIR / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata → {meta_path}")
    print(f"Strategy distribution: {dict(sorted(overall_dist.items()))}")


# ---------------------------------------------------------------------------
# Step 4: Regenerate headlines with article context via step-3.5-flash
# ---------------------------------------------------------------------------

GENERATE_OUTPUT_FILE = PROCESSED_DIR / "sarcasm_pairs_sar_to_non_context_enhanced.jsonl"

# Model with vision capabilities for image-only articles
GENERATE_MODEL = "qwen/qwen3.6-plus:free"
GENERATE_MAX_WORKERS = 30
GENERATE_RATE_LIMIT_PER_MINUTE = 60
GENERATE_RATE_LIMIT_DELAY = 60.0 / GENERATE_RATE_LIMIT_PER_MINUTE
GENERATE_MAX_RETRIES = 5

GENERATE_SYSTEM_PROMPT = """You are a sarcasm style transfer expert for academic NLP research.

TASK: Convert a sarcastic headline into a straightforward, non-sarcastic news headline.
You are given the headline AND the actual article context (body text and/or featured image).

Use the article context to understand what really happened, then rewrite 
a headline that captures the real meaning. Rewrite based on what the article 
is actually about. The output should read like a real, neutral news headline from a mainstream outlet.

For image-only articles (memes/satire images), use the image to understand the satirical point
and write a headline that captures the real meaning.

Sarcasm strategies to identify: sarcasm, irony, satire, understatement, overstatement, rhetorical_question

Output format (JSON only):
{"output": "rewritten non-sarcastic headline", "strategy": "sarcasm"}

No explanations outside the JSON. Output valid JSON only."""


def step_generate() -> None:
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import as_completed as futures_as_completed

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    print("=== Step 4: Regenerate headlines with article context ===")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Load CV-filtered pairs
    if not FILTERED_FILE.exists():
        print(f"ERROR: {FILTERED_FILE} not found. Run --step filter first.")
        return
    pairs = load_jsonl(FILTERED_FILE)

    # Load scrape cache to get article bodies
    cache: dict[str, dict] = {}
    if SCRAPE_CACHE_FILE.exists():
        for r in load_jsonl(SCRAPE_CACHE_FILE):
            cache[r["url"]] = r
    print(f"Loaded {len(cache)} scrape results")

    # Load already-generated headlines for resume
    done: set[str] = set()
    if GENERATE_OUTPUT_FILE.exists():
        for r in load_jsonl(GENERATE_OUTPUT_FILE):
            done.add(r["original_headline"])
    print(f"Already generated: {len(done)}")

    # Build items list — skip if no body and no image (nothing for the model to use)
    items = []
    skipped_no_content = 0
    for p in pairs:
        if p["original_headline"] in done:
            continue
        scrape = cache.get(p["article_link"], {})
        body = scrape.get("body_text", "")
        image_url = scrape.get("image_url", "")
        if not body and not image_url:
            skipped_no_content += 1
            continue
        items.append(
            {
                "original_headline": p["original_headline"],
                "article_link": p["article_link"],
                "strategy": p.get("strategy", ""),
                "body_text": body,
                "image_url": image_url,
            }
        )

    print(f"To generate: {len(items)} (skipped {skipped_no_content} with no content)")
    print(f"Model: {GENERATE_MODEL} (vision-capable)")
    if not items:
        print("Nothing to generate!")
        return

    # Thread-safe locks and shared state
    file_lock = threading.Lock()
    progress_lock = threading.Lock()
    last_request_time = [0.0]
    total_processed = [0]
    total_errors = [0]

    def rate_limited_call(messages):
        """Make a rate-limited request to OpenRouter."""
        with progress_lock:
            now = time.time()
            wait = GENERATE_RATE_LIMIT_DELAY - (now - last_request_time[0])
            if wait > 0:
                time.sleep(wait)
            last_request_time[0] = time.time()
        return client.chat.completions.create(
            model=GENERATE_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

    def build_user_message(item: dict) -> list[dict]:
        """Build multimodal user message content with text + image_url parts."""
        content_parts: list[dict] = []

        # Text part: headline + body context
        if item["body_text"]:
            ctx = item["body_text"]
            text = f'HEADLINE: "{item["original_headline"]}"\nCONTEXT: {ctx}'
        else:
            text = f'HEADLINE: "{item["original_headline"]}"\nCONTEXT: [see attached image]'
        content_parts.append({"type": "text", "text": text})

        # Image part: attach featured image if available
        if item["image_url"]:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item["image_url"]},
                }
            )

        return content_parts

    def process_single(item: dict, item_id: int) -> dict | None:
        """Process a single headline through the model."""
        user_content = build_user_message(item)
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        for attempt in range(GENERATE_MAX_RETRIES):
            try:
                response = rate_limited_call(messages)
                content = response.choices[0].message.content

                # Check for content filter
                if response.choices[0].finish_reason == "content_filter":
                    print(
                        f"    Item {item_id}: Content filtered, retrying...", flush=True
                    )
                    if attempt < GENERATE_MAX_RETRIES - 1:
                        time.sleep(5)
                        continue
                    else:
                        print(
                            f"    Item {item_id}: Content filtered after retries",
                            flush=True,
                        )
                        return None

                # Parse JSON (with fallbacks for markdown-wrapped responses)
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    for delim in ("```json", "```"):
                        if delim in content:
                            content = content.split(delim)[1].split("```")[0].strip()
                            break
                    else:
                        start, end = content.find("{"), content.rfind("}")
                        if start != -1:
                            content = content[start : end + 1]
                    parsed = json.loads(content)

                return {
                    "original_headline": item["original_headline"],
                    "generated_headline": parsed.get("output", ""),
                    "strategy": parsed.get("strategy", item["strategy"]),
                    "type": "sarcastic_to_non",
                    "model_used": GENERATE_MODEL,
                    "article_link": item["article_link"],
                    "context_used": "body" if item["body_text"] else "image_url",
                }

            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["rate limit", "429", "quota"]):
                    wait_time = 5 * (attempt + 1)
                    print(
                        f"    Rate limited on item {item_id}. Waiting {wait_time}s...",
                        flush=True,
                    )
                    time.sleep(wait_time)
                    if attempt < GENERATE_MAX_RETRIES - 1:
                        continue
                if attempt < GENERATE_MAX_RETRIES - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    print(
                        f"    Item {item_id} failed after {GENERATE_MAX_RETRIES} attempts: {e}",
                        flush=True,
                    )
                    return None
        return None

    def worker_thread(worker_id: int, worker_items: list[tuple[int, dict]]):
        """Worker processes multiple items sequentially."""
        for item_id, item in worker_items:
            result = process_single(item, item_id)

            if result:
                with file_lock:
                    with open(GENERATE_OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                with progress_lock:
                    total_processed[0] += 1
                    slug = item["original_headline"][:50]
                    print(
                        f"Worker {worker_id}: [{total_processed[0]}/{len(items)}] "
                        f"{slug}...",
                        flush=True,
                    )
            else:
                with progress_lock:
                    total_errors[0] += 1
                    print(
                        f"Worker {worker_id}: Item {item_id} failed. "
                        f"Errors: {total_errors[0]}",
                        flush=True,
                    )

    total_items = len(items)
    est_minutes = (total_items * GENERATE_RATE_LIMIT_DELAY) / 60
    print("\nProcessing Plan:")
    print(f"   Total items: {total_items}")
    print(
        f"   Rate limit: {GENERATE_RATE_LIMIT_PER_MINUTE} req/min "
        f"({GENERATE_RATE_LIMIT_DELAY:.1f}s between requests)"
    )
    print(f"   Workers: {GENERATE_MAX_WORKERS}")
    print(f"   Est. time: ~{est_minutes:.1f} minutes")
    print(f"   Model: {GENERATE_MODEL}")
    print("\nStarting processing...\n", flush=True)

    GENERATE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Distribute items round-robin across workers
    per_worker: list[list[tuple[int, dict]]] = [[] for _ in range(GENERATE_MAX_WORKERS)]
    for i, item in enumerate(items):
        per_worker[i % GENERATE_MAX_WORKERS].append((i, item))

    with ThreadPoolExecutor(max_workers=GENERATE_MAX_WORKERS) as executor:
        futures = []
        for worker_id, wi in enumerate(per_worker):
            if wi:
                futures.append(executor.submit(worker_thread, worker_id, wi))

        for f in futures_as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Worker error: {e}", flush=True)

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"Total pairs generated: {total_processed[0]}")
    print(f"Errors: {total_errors[0]}")
    print(f"Output: {GENERATE_OUTPUT_FILE}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        choices=["filter", "scrape", "split", "generate", "all"],
        default="all",
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step in ("filter", "all"):
        step_filter()

    if args.step in ("scrape", "all"):
        step_scrape()

    if args.step in ("split", "all"):
        step_split()

    if args.step == "generate":
        step_generate()


if __name__ == "__main__":
    main()
