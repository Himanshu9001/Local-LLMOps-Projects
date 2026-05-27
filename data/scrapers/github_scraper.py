"""
GitHub postmortem scraper — fetches real incident reports from curated repos
Outputs: data/raw/github_postmortems_raw.jsonl
Sources:
  - danluu/post-mortems (curated list of public postmortems)
  - raw markdown files from GitHub API
Requires: GITHUB_TOKEN env var (free, needed for rate limit: 5000 req/hr vs 60)
"""

import requests
import json
import time
import os
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OUTPUT_FILE  = Path("data/raw/github_postmortems_raw.jsonl")
STATE_FILE   = Path("data/raw/.github_visited.json")

# Curated repos with postmortem/incident content
REPOS = [
    {"owner": "danluu",        "repo": "post-mortems",         "path": ""},
    {"owner": "mtdvio",        "repo": "every-programmer-should-know", "path": ""},
]

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "research-bot/1.0",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

DELAY_SEC = 1.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state() -> set:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()))
    return set()

def save_state(visited: set):
    STATE_FILE.write_text(json.dumps(list(visited)))

def get_repo_files(owner: str, repo: str, path: str = "") -> list[dict]:
    """
    Recursively list all .md files in a GitHub repo using the Contents API.
    Returns list of {name, download_url, path} dicts.
    """
    url  = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, headers=HEADERS, timeout=15)

    if resp.status_code == 403:
        print(f"  [RATE LIMIT] {owner}/{repo} — add GITHUB_TOKEN env var")
        return []
    if resp.status_code != 200:
        return []

    items = resp.json()
    files = []

    for item in items:
        if item["type"] == "file" and item["name"].endswith(".md"):
            files.append(item)
        elif item["type"] == "dir":
            # Recurse into subdirectories
            time.sleep(0.3)
            files.extend(get_repo_files(owner, repo, item["path"]))

    return files

def fetch_markdown_content(download_url: str) -> str | None:
    """Download raw markdown content from GitHub."""
    try:
        resp = requests.get(download_url, headers=HEADERS, timeout=15)
        return resp.text if resp.status_code == 200 else None
    except:
        return None

def parse_postmortem(text: str) -> list[dict]:
    """
    danluu/post-mortems actual format:
      [Company](url). Description sentence(s).
    No list prefix — plain paragraph links followed by a period and description.
    """
    import re
    records = []

    # Match: [Company](url). Description
    pattern = re.compile(r'\[([^\]]+)\]\((https?://[^)]+)\)\.\s*(.+?)(?=\n\[|\n\n|\Z)', re.DOTALL)

    for match in pattern.finditer(text):
        company, url, description = match.groups()
        description = description.strip().replace('\n', ' ')
        if len(description) < 20:
            continue
        records.append({
            "source":  "github_postmortem",
            "title":   company,
            "url":     url,
            "content": f"Company: {company}\n\nIncident Summary: {description}",
            "type":    "postmortem_reference"
        })
    return records

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    visited = load_state()
    total   = 0

    with open(OUTPUT_FILE, "a") as f:
        for repo_config in REPOS:
            owner = repo_config["owner"]
            repo  = repo_config["repo"]
            print(f"[github_scraper] Processing {owner}/{repo}")

            files = get_repo_files(owner, repo, repo_config["path"])
            print(f"  Found {len(files)} markdown files")

            for file_info in files:
                file_id = f"{owner}/{repo}/{file_info['path']}"
                if file_id in visited:
                    continue

                content = fetch_markdown_content(file_info["download_url"])
                if not content:
                    visited.add(file_id)
                    continue

                # Special parsing for danluu/post-mortems index
                if owner == "danluu" and repo == "post-mortems":
                    records = parse_postmortem(content)
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
                    total += len(records)
                    print(f"  {file_info['name']}: {len(records)} postmortem references")
                else:
                    # Generic markdown file — store as-is for Q&A generation later
                    doc = {
                        "source":  f"github_{owner}_{repo}",
                        "url":     file_info.get("html_url", ""),
                        "title":   file_info["name"].replace(".md", ""),
                        "content": content
                    }
                    if len(content) > 200:
                        f.write(json.dumps(doc) + "\n")
                        total += 1

                visited.add(file_id)
                time.sleep(DELAY_SEC)

            save_state(visited)

    print(f"[github_scraper] Done. {total} records → {OUTPUT_FILE}")

if __name__ == "__main__":
    run()