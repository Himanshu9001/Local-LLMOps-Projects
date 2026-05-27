"""
AWS docs scraper — crawls docs.aws.amazon.com for EKS, CloudWatch, IAM, Terraform
Outputs: data/raw/aws_raw.jsonl
Strategy: Sitemap-driven (AWS publishes per-service sitemaps)
"""

import requests
import json
import time
from bs4 import BeautifulSoup
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# AWS publishes per-service sitemaps — more reliable than crawling
SERVICE_SITEMAPS = [
    "https://docs.aws.amazon.com/eks/latest/userguide/sitemap.xml",
    "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/sitemap.xml",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/sitemap.xml",
    "https://docs.aws.amazon.com/cli/latest/userguide/sitemap.xml",
]
OUTPUT_FILE = Path("data/raw/aws_raw.jsonl")
STATE_FILE  = Path("data/raw/.aws_visited.json")
MAX_PAGES   = 1500
DELAY_SEC   = 1.5
HEADERS     = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state() -> set:
    if Path(STATE_FILE).exists():
        return set(json.loads(Path(STATE_FILE).read_text()))
    return set()

def save_state(visited: set):
    Path(STATE_FILE).write_text(json.dumps(list(visited)))

def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Parse XML sitemap and return all <loc> URLs."""
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "xml")
        return [loc.text for loc in soup.find_all("loc")]
    except Exception as e:
        print(f"  [SITEMAP ERROR] {sitemap_url}: {e}")
        return []

def extract_content(url: str) -> dict | None:
    """
    AWS docs use <div id="main-content"> or <div class="awsdocs-container">.
    Strip left nav and right feedback panel.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # AWS docs main content
        content_tag = (
            soup.find("div", id="main-content") or
            soup.find("div", class_="awsdocs-container") or
            soup.find("main")
        )
        if not content_tag:
            return None

        for tag in content_tag.find_all(["nav", "aside", "script", "style"]):
            tag.decompose()

        text = content_tag.get_text(separator="\n", strip=True)
        if len(text) < 200:
            return None

        return {"source": "aws_docs", "url": url, "title": title, "content": text}

    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    visited = load_state()

    # Collect all URLs across all service sitemaps
    all_urls = []
    for sitemap in SERVICE_SITEMAPS:
        urls = get_urls_from_sitemap(sitemap)
        print(f"[aws_scraper] {sitemap.split('/')[4]}: {len(urls)} URLs")
        all_urls.extend(urls)

    pending = [u for u in all_urls if u not in visited][:MAX_PAGES]
    print(f"[aws_scraper] {len(pending)} URLs to scrape")

    with open(OUTPUT_FILE, "a") as f:
        for i, url in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {url}")
            doc = extract_content(url)
            if doc:
                f.write(json.dumps(doc) + "\n")

            visited.add(url)
            if i % 50 == 0:
                save_state(visited)

            time.sleep(DELAY_SEC)

    save_state(visited)
    print(f"[aws_scraper] Done → {OUTPUT_FILE}")

if __name__ == "__main__":
    run()