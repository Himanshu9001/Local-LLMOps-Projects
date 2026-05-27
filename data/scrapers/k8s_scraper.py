"""
Kubernetes docs scraper — crawls kubernetes.io/docs
Outputs: data/raw/k8s_raw.jsonl
Strategy: Sitemap-driven URL discovery → BeautifulSoup content extraction
Resumable: tracks visited URLs in a local state file
"""

import requests
import json
import time
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://kubernetes.io/docs/"
SITEMAP_URL = "https://kubernetes.io/sitemap.xml"
OUTPUT_FILE = Path("data/raw/k8s_raw.jsonl")
STATE_FILE  = Path("data/raw/.k8s_visited.json")   # resumability checkpoint
MAX_PAGES   = 2000
DELAY_SEC   = 1.5                                   # polite crawl delay
HEADERS     = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state() -> set:
    """Load already-visited URLs from checkpoint file."""
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()

def save_state(visited: set):
    """Persist visited URLs so the scraper can resume after interruption."""
    STATE_FILE.write_text(json.dumps(list(visited)))

def get_doc_urls_from_sitemap() -> list[str]:
    """
    kubernetes.io returns a sitemap INDEX — a list of per-language sitemaps.
    Step 1: Parse the index to find the English sitemap URL.
    Step 2: Parse the English sitemap to get actual /docs/ page URLs.
    """
    # Step 1 — fetch sitemap index
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=15)
    index_soup = BeautifulSoup(resp.text, "xml")

    # Extract only the English sitemap — skip bn, zh-cn, fr, etc.
    english_sitemap = None
    for loc in index_soup.find_all("loc"):
        if "/en/sitemap.xml" in loc.text:
            english_sitemap = loc.text
            break

    if not english_sitemap:
        print("[k8s_scraper] ERROR: Could not find English sitemap in index")
        return []

    print(f"[k8s_scraper] Found English sitemap: {english_sitemap}")

    # Step 2 — fetch actual English sitemap and extract /docs/ URLs
    resp2 = requests.get(english_sitemap, headers=HEADERS, timeout=15)
    soup2 = BeautifulSoup(resp2.text, "xml")

    urls = [
        loc.text for loc in soup2.find_all("loc")
        if "/docs/" in loc.text and not loc.text.endswith(".pdf")
    ]
    print(f"[k8s_scraper] Found {len(urls)} /docs/ URLs")
    return urls

def extract_content(url: str) -> dict | None:
    """
    Fetch a single doc page and extract:
      - title (h1)
      - main content text (article or main tag)
      - source URL
    Returns None on fetch failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract main content — kubernetes.io uses <article> or <div class="td-content">
        content_tag = (
            soup.find("article") or
            soup.find("div", class_="td-content") or
            soup.find("main")
        )
        if not content_tag:
            return None

        # Strip nav, TOC, footer noise
        for tag in content_tag.find_all(["nav", "aside", "footer", "script", "style"]):
            tag.decompose()

        text = content_tag.get_text(separator="\n", strip=True)

        # Skip pages with very little content (nav-only pages)
        if len(text) < 200:
            return None

        return {
            "source":  "kubernetes_docs",
            "url":     url,
            "title":   title,
            "content": text
        }

    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint — skip already scraped URLs
    visited = load_state()
    print(f"[k8s_scraper] Resuming — {len(visited)} URLs already visited")

    # Discover URLs from sitemap
    all_urls = get_doc_urls_from_sitemap()
    pending  = [u for u in all_urls if u not in visited][:MAX_PAGES]
    print(f"[k8s_scraper] {len(pending)} URLs to scrape")

    # Append mode — safe for resuming
    with open(OUTPUT_FILE, "a") as f:
        for i, url in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {url}")

            doc = extract_content(url)
            if doc:
                f.write(json.dumps(doc) + "\n")

            visited.add(url)

            # Save checkpoint every 50 pages
            if i % 50 == 0:
                save_state(visited)

            time.sleep(DELAY_SEC)   # rate limiting — be polite

    save_state(visited)
    print(f"[k8s_scraper] Done. Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()