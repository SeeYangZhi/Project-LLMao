"""Scrape article headlines from The Onion's /latest/ archive.

Paginates through https://theonion.com/latest/page/{N}/ and extracts
headline text + article URL from each page. Outputs a JSONL file.

Usage:
    uv run python scripts/data_prep/scrape_onion_headlines.py
    uv run python scripts/data_prep/scrape_onion_headlines.py --pages 60 --output data/raw/onion_headlines.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "onion_headlines.jsonl"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# The Onion /latest/ serves ~24 articles per page.
# 45 pages * 24 ≈ 1,080 headlines — comfortably over 1,000.
DEFAULT_PAGES = 45
REQUEST_DELAY = 1.5  # seconds between requests to be polite
MAX_RETRIES = 3


def fetch_page(url: str, timeout: int = 15) -> str | None:
    """Fetch a single page, returning HTML or None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # past last page
            print(f"  HTTP {e.code} on {url} (attempt {attempt + 1})")
        except Exception as e:
            print(f"  Error on {url} (attempt {attempt + 1}): {e}")
        time.sleep(2 ** attempt)
    return None


def extract_headlines(html: str) -> list[dict]:
    """Extract headline text and URL from a /latest/ page.

    Headlines live in <h3 class="...wp-block-post-title..."> tags,
    each containing an <a> linking to the article.
    """
    # Match h3 tags with wp-block-post-title class containing an anchor
    pattern = re.compile(
        r'<h3[^>]*wp-block-post-title[^>]*>\s*'
        r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    results = []
    for match in pattern.finditer(html):
        url = match.group(1).strip()
        headline = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        headline = re.sub(r"\s+", " ", headline)
        if headline:
            results.append({"headline": headline, "url": url})
    return results


def main():
    parser = argparse.ArgumentParser(description="Scrape Onion headlines from /latest/")
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGES,
        help=f"Number of pages to scrape (default: {DEFAULT_PAGES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Page number to start from (default: 1, for resuming)",
    )
    args = parser.parse_args()

    # Load existing headlines if resuming
    seen_urls: set[str] = set()
    existing: list[dict] = []
    if args.start_page > 1 and args.output.exists():
        with open(args.output) as f:
            for line in f:
                rec = json.loads(line)
                existing.append(rec)
                seen_urls.add(rec["url"])
        print(f"Loaded {len(existing)} existing headlines from {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_headlines = list(existing)
    end_page = args.start_page + args.pages

    print(f"Scraping The Onion /latest/ pages {args.start_page}–{end_page - 1}")
    print(f"Output: {args.output}\n")

    for page_num in range(args.start_page, end_page):
        if page_num == 1:
            url = "https://theonion.com/latest/"
        else:
            url = f"https://theonion.com/latest/page/{page_num}/"

        html = fetch_page(url)
        if html is None:
            print(f"  Page {page_num}: failed or 404 — stopping.")
            break

        headlines = extract_headlines(html)
        new_count = 0
        for h in headlines:
            if h["url"] not in seen_urls:
                seen_urls.add(h["url"])
                all_headlines.append(h)
                new_count += 1

        print(
            f"  Page {page_num}: {len(headlines)} found, {new_count} new "
            f"(total: {len(all_headlines)})"
        )

        if page_num < end_page - 1:
            time.sleep(REQUEST_DELAY)

    # Write all headlines
    with open(args.output, "w") as f:
        for h in all_headlines:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_headlines)} unique headlines saved to {args.output}")


if __name__ == "__main__":
    main()
