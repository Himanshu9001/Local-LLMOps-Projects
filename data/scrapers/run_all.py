"""
Master runner — executes all scrapers sequentially.
Run: python data/scrapers/run_all.py
Each scraper is resumable independently via its own state file.
"""

import subprocess
import sys
from pathlib import Path

SCRAPERS = [
    "data/scrapers/k8s_scraper.py",
    "data/scrapers/azure_scraper.py",
    "data/scrapers/aws_scraper.py",
    "data/scrapers/github_scraper.py",
    "data/scrapers/stackoverflow_scraper.py",
]

def run():
    for scraper in SCRAPERS:
        print(f"\n{'='*60}")
        print(f"Running: {scraper}")
        print('='*60)
        result = subprocess.run(
            [sys.executable, scraper],
            capture_output=False   # stream output live
        )
        if result.returncode != 0:
            print(f"[WARNING] {scraper} exited with code {result.returncode}")
            # Continue to next scraper — don't abort the whole run

if __name__ == "__main__":
    run()