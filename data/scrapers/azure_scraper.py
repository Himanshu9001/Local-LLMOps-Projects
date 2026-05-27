"""
Azure docs scraper — crawls learn.microsoft.com for AKS, Terraform, Monitor
Outputs: data/raw/azure_raw.jsonl
Strategy: Predefined topic URL seeds → recursive same-domain link following
"""

import requests
import json
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
SEED_URLS = [
    "https://learn.microsoft.com/en-us/azure/aks/",
    "https://learn.microsoft.com/en-us/azure/developer/terraform/",
    "https://learn.microsoft.com/en-us/azure/azure-monitor/",
    "https://learn.microsoft.com/en-us/azure/devops/",
]
OUTPUT_FILE = Path("data/raw/azure_raw.jsonl")
STATE_FILE  = Path("data/raw/.azure_visited.json")
MAX_PAGES   = 1500
DELAY_SEC   = 1.5
HEADERS     = {
    "User-Agent": "Mozilla/5.0 (research-bot/1.0)",
    # Azure docs require accepting cookies to get full content
    "Cookie": "MSCC=; MC1=;"
}
ALLOWED_DOMAIN = "learn.microsoft.com"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state() -> set:
    if Path(STATE_FILE).exists():
        return set(json.loads(Path(STATE_FILE).read_text()))
    return set()

def save_state(visited: set):
    Path(STATE_FILE).write_text(json.dumps(list(visited)))

def is_valid_url(url: str) -> bool:
    """Only follow URLs within Azure docs domain, skip anchors and non-HTML."""
    parsed = urlparse(url)
    return (
        parsed.netloc == ALLOWED_DOMAIN and
        "/en-us/azure/" in parsed.path and
        not url.endswith((".pdf", ".zip", ".png", ".jpg")) and
        "#" not in url
    )

def extract_content(url: str, soup: BeautifulSoup) -> dict | None:
    """
    Extract title + main content from Azure learn pages.
    Azure uses <main id="main"> for primary content.
    """
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Azure docs main content container
    content_tag = soup.find("main", id="main") or soup.find("div", class_="content")
    if not content_tag:
        return None

    for tag in content_tag.find_all(["nav", "aside", "footer", "script", "style", "div", "section"],
                                     class_=["feedback", "breadcrumb", "toc", "nav"]):
        tag.decompose()

    text = content_tag.get_text(separator="\n", strip=True)
    if len(text) < 200:
        return None

    return {"source": "azure_docs", "url": url, "title": title, "content": text}

def get_child_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    """Extract all valid same-domain links from a page for BFS crawling."""
    links = []
    for a in soup.find_all("a", href=True):
        full_url = urljoin(current_url, a["href"]).split("#")[0]  # strip anchors
        if is_valid_url(full_url):
            links.append(full_url)
    return links

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    visited = load_state()
    queue   = deque([u for u in SEED_URLS if u not in visited])
    count   = 0

    print(f"[azure_scraper] Resuming — {len(visited)} URLs already visited")

    with open(OUTPUT_FILE, "a") as f:
        while queue and count < MAX_PAGES:
            url = queue.popleft()
            if url in visited:
                continue

            print(f"  [{count+1}] {url}")
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                if resp.status_code != 200:
                    visited.add(url)
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                doc  = extract_content(url, soup)

                if doc:
                    f.write(json.dumps(doc) + "\n")
                    # BFS — add child links discovered on this page
                    for link in get_child_links(soup, url):
                        if link not in visited:
                            queue.append(link)

                visited.add(url)
                count += 1

                if count % 50 == 0:
                    save_state(visited)

                time.sleep(DELAY_SEC)

            except Exception as e:
                print(f"  [ERROR] {url}: {e}")
                visited.add(url)

    save_state(visited)
    print(f"[azure_scraper] Done. {count} pages scraped → {OUTPUT_FILE}")

if __name__ == "__main__":
    run()