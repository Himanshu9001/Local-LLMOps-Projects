"""
Phase 2 — Data Cleaning + Dataset Preparation
Reads all raw JSONL files → cleans → generates Q&A pairs → outputs train.jsonl

Pipeline:
  1. Clean + deduplicate all sources
  2. SO pairs → direct format conversion (already Q&A)
  3. Doc pages → rule-based Q&A extraction (heading + first paragraph)
  4. Postmortems → structured RCA Q&A template
  5. Merge + shuffle → data/processed/train.jsonl

Output format: Mistral instruct format
  <s>[INST] system + question [/INST] answer </s>
"""

import json
import re
import random
import hashlib
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE   = PROCESSED_DIR / "train.jsonl"
STATS_FILE    = PROCESSED_DIR / "dataset_stats.json"

# Filtering thresholds
MIN_CONTENT_LEN  = 200    # chars — skip stub pages
MAX_CONTENT_LEN  = 50000  # chars — skip massive pages (likely auto-generated)
MIN_ANSWER_LEN   = 100    # chars — skip trivial answers
MAX_ANSWER_LEN   = 2000   # chars — keep answers concise for training

SYSTEM_PROMPT = (
    "You are an expert SRE and DevOps engineer with deep knowledge of "
    "Kubernetes, Azure, AWS, Terraform, CI/CD pipelines, and incident management. "
    "Provide clear, actionable answers with commands where relevant."
)

SEED = 42
random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts. Skip malformed lines."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [SKIP] Malformed JSON at line {i} in {path.name}")
    return records

def clean_text(text: str) -> str:
    """
    Normalize whitespace, remove zero-width chars, collapse blank lines.
    Preserves code blocks and commands — critical for SRE content.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # control chars
    text = re.sub(r'[ \t]+', ' ', text)           # collapse horizontal whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)        # max 2 consecutive blank lines
    return text.strip()

def content_hash(text: str) -> str:
    """MD5 hash of first 500 chars — fast deduplication fingerprint."""
    return hashlib.md5(text[:500].encode()).hexdigest()

def format_training_example(question: str, answer: str) -> dict:
    """
    Wrap Q&A into Mistral instruct format.
    <s>[INST] system\\n\\nquestion [/INST] answer </s>
    This is the exact format Mistral-7B-Instruct-v0.3 was trained on.
    """
    text = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"{question.strip()} [/INST] "
        f"{answer.strip()} </s>"
    )
    return {"text": text}

def truncate_answer(text: str, max_len: int = MAX_ANSWER_LEN) -> str:
    """
    Truncate answer at sentence boundary to avoid mid-sentence cutoffs.
    Prefers ending at a period, newline, or code block boundary.
    """
    if len(text) <= max_len:
        return text
    # Find last sentence boundary before max_len
    truncated = text[:max_len]
    last_period = max(truncated.rfind('. '), truncated.rfind('\n'))
    if last_period > max_len * 0.7:   # only truncate at boundary if it's not too short
        return truncated[:last_period + 1].strip()
    return truncated.strip()

# ── Source 1: Stack Overflow ──────────────────────────────────────────────────

def process_stackoverflow(records: list[dict]) -> list[dict]:
    """
    SO records are already Q&A pairs.
    question = title + body (if body adds context)
    answer   = best answer body
    Filter: min score thresholds already applied in scraper.
    """
    examples = []
    seen     = set()

    for rec in records:
        title   = clean_text(rec.get("title", ""))
        q_body  = clean_text(rec.get("question_body", ""))
        answer  = clean_text(rec.get("answer_body", ""))

        if not title or not answer:
            continue
        if len(answer) < MIN_ANSWER_LEN:
            continue

        # Build question — include body only if it adds meaningful context
        if len(q_body) > 100 and q_body.lower() != title.lower():
            question = f"{title}\n\n{q_body[:500]}"  # cap body to 500 chars
        else:
            question = title

        answer = truncate_answer(answer)

        # Deduplicate by question hash
        h = content_hash(question)
        if h in seen:
            continue
        seen.add(h)

        # Add source tag to answer for traceability
        tags_str = ", ".join(rec.get("tags", []))
        answer_with_source = f"{answer}\n\n[Source: Stack Overflow | Tags: {tags_str}]"

        examples.append(format_training_example(question, answer_with_source))

    print(f"  [SO] {len(examples)} training examples")
    return examples

# ── Source 2: Documentation Pages (K8s, Azure, AWS) ──────────────────────────

def chunk_document(content: str, chunk_size: int = 1500) -> list[str]:
    """
    Split long doc pages into chunks at paragraph boundaries.
    chunk_size in chars — balances context vs token limit.
    Splitting at paragraphs preserves semantic coherence.
    """
    paragraphs = content.split('\n\n')
    chunks     = []
    current    = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > chunk_size and current:
            chunks.append('\n\n'.join(current))
            current     = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append('\n\n'.join(current))

    return chunks

def extract_qa_from_chunk(chunk: str, title: str, source: str) -> dict | None:
    """
    Rule-based Q&A extraction from a doc chunk.
    Strategy:
      - Question: derive from the chunk's first heading or the page title
      - Answer:   the chunk content itself (already explanatory text)
    This works well for docs because headings are naturally question-like.
    """
    lines = chunk.strip().split('\n')

    # Find first heading in chunk (markdown # or ALL CAPS line)
    heading = None
    content_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match(r'^#{1,3}\s+', line):
            heading = re.sub(r'^#{1,3}\s+', '', line).strip()
            content_start = i + 1
            break
        elif len(line) > 10 and line.isupper():
            heading = line.title()
            content_start = i + 1
            break

    # Fall back to page title if no heading found
    if not heading:
        heading = title

    # Build answer from content after heading
    answer_lines = [l for l in lines[content_start:] if l.strip()]
    answer = '\n'.join(answer_lines).strip()

    if len(answer) < MIN_ANSWER_LEN:
        return None

    answer = truncate_answer(answer)

    # Convert heading to a natural question
    question = heading_to_question(heading, source)
    answer_with_source = f"{answer}\n\n[Source: {source}]"

    return format_training_example(question, answer_with_source)

def heading_to_question(heading: str, source: str) -> str:
    """
    Convert a doc heading into a natural question.
    Examples:
      'Configure RBAC'           → 'How do I configure RBAC?'
      'Pod Security Standards'   → 'What are Pod Security Standards?'
      'Troubleshooting'          → 'How do I troubleshoot issues in Kubernetes?'
    """
    heading = heading.strip().rstrip('?').rstrip('.')

    # Imperative verb patterns → How do I ...?
    imperative_verbs = [
        'configure', 'install', 'deploy', 'set up', 'create', 'enable',
        'disable', 'update', 'upgrade', 'migrate', 'manage', 'monitor',
        'debug', 'troubleshoot', 'fix', 'resolve', 'run', 'use', 'implement'
    ]
    first_word = heading.split()[0].lower() if heading.split() else ""
    if first_word in imperative_verbs:
        return f"How do I {heading.lower()}?"

    # Troubleshooting patterns
    if any(w in heading.lower() for w in ['troubleshoot', 'error', 'issue', 'problem', 'fail', 'debug']):
        return f"How do I troubleshoot: {heading}?"

    # Concept patterns → What is/are ...?
    plural_indicators = ['s', 'policies', 'standards', 'concepts', 'types', 'methods']
    last_word = heading.split()[-1].lower() if heading.split() else ""
    if last_word in plural_indicators or len(heading.split()) <= 3:
        return f"What are {heading}?"

    # Default → explain
    return f"Explain {heading} in the context of {source.replace('_docs', '').replace('_', ' ')}."

def process_docs(records: list[dict], source_name: str) -> list[dict]:
    """
    Process raw doc pages into Q&A training examples.
    Each page → chunked → rule-based Q&A extraction per chunk.
    """
    examples = []
    seen     = set()
    skipped  = 0

    for rec in records:
        content = clean_text(rec.get("content", ""))
        title   = rec.get("title", "")
        source  = rec.get("source", source_name)

        # Length filter
        if len(content) < MIN_CONTENT_LEN or len(content) > MAX_CONTENT_LEN:
            skipped += 1
            continue

        # Chunk the document
        chunks = chunk_document(content)

        for chunk in chunks:
            h = content_hash(chunk)
            if h in seen:
                continue
            seen.add(h)

            example = extract_qa_from_chunk(chunk, title, source)
            if example:
                examples.append(example)

    print(f"  [{source_name}] {len(records)} pages → {len(examples)} examples (skipped {skipped})")
    return examples

# ── Source 3: GitHub Postmortems ──────────────────────────────────────────────

def process_postmortems(records: list[dict]) -> list[dict]:
    """
    Convert each postmortem summary into a structured RCA Q&A pair.
    Question: What happened during the [Company] incident?
    Answer:   Structured RCA format — incident, root cause, impact, lesson
    """
    examples = []
    seen     = set()

    # RCA answer template — teaches the model structured incident analysis
    RCA_TEMPLATE = (
        "Incident Report: {title}\n\n"
        "Summary: {summary}\n\n"
        "Root Cause Analysis:\n"
        "- What happened: {summary}\n"
        "- Likely root cause: Infrastructure or configuration failure\n"
        "- Impact: Service degradation or outage\n"
        "- Remediation: Review the incident at {url}\n\n"
        "Key SRE Lesson: Document all incidents with timeline, impact scope, "
        "root cause, and follow-up action items to prevent recurrence.\n\n"
        "[Source: {source} | URL: {url}]"
    )

    for rec in records:
        title   = rec.get("title", "").strip()
        content = clean_text(rec.get("content", ""))
        url     = rec.get("url", "")
        source  = rec.get("source", "github_postmortem")

        if not title or not content:
            continue

        h = content_hash(content)
        if h in seen:
            continue
        seen.add(h)

        # Extract summary — content is "Company: X\n\nIncident Summary: Y"
        summary_match = re.search(r'Incident Summary:\s*(.+)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else content

        if len(summary) < 30:
            continue

        question = f"What happened during the {title} incident and what was the root cause?"
        answer   = RCA_TEMPLATE.format(
            title=title, summary=summary[:500], url=url, source=source
        )

        examples.append(format_training_example(question, answer))

    print(f"  [Postmortems] {len(examples)} training examples")
    return examples

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_examples = []
    stats        = {}

    print("\n=== Phase 2: Dataset Preparation ===\n")

    # 1. Stack Overflow
    print("[1/4] Processing Stack Overflow...")
    so_records = load_jsonl(RAW_DIR / "stackoverflow_raw.jsonl")
    so_examples = process_stackoverflow(so_records)
    all_examples.extend(so_examples)
    stats["stackoverflow"] = len(so_examples)

    # 2. Kubernetes docs
    print("\n[2/4] Processing Kubernetes docs...")
    k8s_records  = load_jsonl(RAW_DIR / "k8s_raw.jsonl")
    k8s_examples = process_docs(k8s_records, "kubernetes_docs")
    all_examples.extend(k8s_examples)
    stats["kubernetes"] = len(k8s_examples)

    # 3. Azure docs
    print("\n[3/4] Processing Azure + AWS docs...")
    azure_records  = load_jsonl(RAW_DIR / "azure_raw.jsonl")
    azure_examples = process_docs(azure_records, "azure_docs")
    all_examples.extend(azure_examples)
    stats["azure"] = len(azure_examples)

    aws_records  = load_jsonl(RAW_DIR / "aws_raw.jsonl")
    aws_examples = process_docs(aws_records, "aws_docs")
    all_examples.extend(aws_examples)
    stats["aws"] = len(aws_examples)

    # 4. GitHub postmortems
    print("\n[4/4] Processing GitHub postmortems...")
    pm_records  = load_jsonl(RAW_DIR / "github_postmortems_raw.jsonl")
    pm_examples = process_postmortems(pm_records)
    all_examples.extend(pm_examples)
    stats["postmortems"] = len(pm_examples)

    # Shuffle — prevent source ordering bias during training
    random.shuffle(all_examples)

    # Write final training file
    with open(OUTPUT_FILE, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    stats["total"] = len(all_examples)

    # Write stats
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset ready: {OUTPUT_FILE}")
    print(f"Total training examples: {len(all_examples)}")
    print(f"\nBreakdown:")
    for source, count in stats.items():
        if source != "total":
            pct = count / len(all_examples) * 100
            print(f"  {source:<20} {count:>5} examples ({pct:.1f}%)")
    print(f"{'='*50}")
    print(f"\nOutput size: ", end="")
    import os
    print(f"{os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    run()