"""
Stack Overflow scraper — fetches high-quality DevOps/SRE Q&A via Stack Exchange API
Outputs: data/raw/stackoverflow_raw.jsonl
Tags targeted: kubernetes, docker, terraform, ansible, github-actions, azure-devops
Already in Q&A format — minimal cleaning needed for training
Requires: SE_API_KEY env var (free at stackapps.com — raises quota 300→10000/day)
"""

import requests
import json
import time
import os
import html
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY     = os.environ.get("SE_API_KEY", "")
OUTPUT_FILE = Path("data/raw/stackoverflow_raw.jsonl")
STATE_FILE  = Path("data/raw/.stackoverflow_state.json")

# High-value DevOps/SRE tags — each tag fetches top-voted questions
TARGET_TAGS = [
    "kubernetes", "docker", "terraform", "github-actions",
    "ansible", "azure-devops", "helm", "prometheus",
    "argo-cd", "jenkins", "nginx-ingress", "azure-kubernetes-service"
]

# Minimum score threshold — ensures quality answers only
MIN_QUESTION_SCORE = 5
MIN_ANSWER_SCORE   = 3
MAX_PER_TAG        = 500
DELAY_SEC          = 0.5   # SE API allows 30 req/sec with key

BASE_URL = "https://api.stackexchange.com/2.3"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state() -> dict:
    """State tracks {tag: last_page_fetched} for resumability."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state))

def clean_html(text: str) -> str:
    """Strip HTML tags and unescape HTML entities from SO content."""
    import re
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)  # preserve code blocks
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()

def fetch_questions(tag: str, page: int) -> dict:
    """
    Fetch paginated questions for a tag via Stack Exchange API.
    sorted=votes ensures highest-quality questions first.
    """
    params = {
        "order":    "desc",
        "sort":     "votes",
        "tagged":   tag,
        "site":     "stackoverflow",
        "filter":   "withbody",        # include question body
        "pagesize": 100,
        "page":     page,
    }
    if API_KEY:
        params["key"] = API_KEY

    resp = requests.get(f"{BASE_URL}/questions", params=params, timeout=15)
    return resp.json()

def fetch_answers(question_id: int) -> list[dict]:
    """Fetch accepted + top answers for a question."""
    params = {
        "order":  "desc",
        "sort":   "votes",
        "site":   "stackoverflow",
        "filter": "withbody",
    }
    if API_KEY:
        params["key"] = API_KEY

    resp = requests.get(f"{BASE_URL}/questions/{question_id}/answers", params=params, timeout=15)
    data = resp.json()
    return data.get("items", [])

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = load_state()
    total = 0

    with open(OUTPUT_FILE, "a") as f:
        for tag in TARGET_TAGS:
            print(f"[so_scraper] Tag: {tag}")
            page       = state.get(tag, 1)
            tag_count  = 0

            while tag_count < MAX_PER_TAG:
                data = fetch_questions(tag, page)
                questions = data.get("items", [])

                if not questions:
                    break

                for q in questions:
                    if q.get("score", 0) < MIN_QUESTION_SCORE:
                        continue
                    if not q.get("is_answered"):
                        continue

                    # Fetch top answers for this question
                    time.sleep(DELAY_SEC)
                    answers = fetch_answers(q["question_id"])

                    # Take the highest-voted answer meeting the score threshold
                    good_answers = [
                        a for a in answers
                        if a.get("score", 0) >= MIN_ANSWER_SCORE
                    ]
                    if not good_answers:
                        continue

                    best_answer = good_answers[0]

                    record = {
                        "source":          "stackoverflow",
                        "question_id":     q["question_id"],
                        "title":           clean_html(q["title"]),
                        "question_body":   clean_html(q.get("body", "")),
                        "answer_body":     clean_html(best_answer.get("body", "")),
                        "tags":            q.get("tags", []),
                        "question_score":  q["score"],
                        "answer_score":    best_answer["score"],
                        "is_accepted":     best_answer.get("is_accepted", False),
                        "url":             q.get("link", "")
                    }
                    f.write(json.dumps(record) + "\n")
                    tag_count += 1
                    total     += 1

                # Pagination
                if not data.get("has_more", False):
                    break

                page += 1
                state[tag] = page
                save_state(state)

                # SE API backoff hint
                if data.get("backoff"):
                    print(f"  [BACKOFF] Waiting {data['backoff']}s")
                    time.sleep(data["backoff"])
                else:
                    time.sleep(DELAY_SEC)

            print(f"  {tag}: {tag_count} Q&A pairs")

    save_state(state)
    print(f"[so_scraper] Done. {total} total records → {OUTPUT_FILE}")

if __name__ == "__main__":
    run()