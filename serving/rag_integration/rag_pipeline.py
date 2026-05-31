"""
Phase 7B — Local RAG Pipeline
pgvector (dense) + BM25 (sparse) hybrid retrieval
Embedding: all-MiniLM-L6-v2 (local, no API)
Storage: PostgreSQL + pgvector (Docker)
"""

import os
import json
import logging
import numpy as np
from typing import Optional
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "sre_rag",
    "user":     "postgres",
    "password": "postgres",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384d, 90MB, runs on CPU
EMBEDDING_DIM   = 384
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K_DENSE     = 10   # pgvector candidates
TOP_K_SPARSE    = 10   # BM25 candidates
TOP_K_FINAL     = 5    # final results after fusion

DATASET_PATH = Path("data/processed/train.jsonl")

# ── Database Setup ─────────────────────────────────────────────────────────────

def get_connection():
    """Get plain PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)

def get_vector_connection():
    """Get PostgreSQL connection with pgvector registered."""
    conn = get_connection()
    register_vector(conn)
    return conn

def setup_database():
    """Create pgvector extension and documents table."""
    conn = get_connection()
    cur  = conn.cursor()

    # Enable pgvector extension FIRST
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)

    # Documents table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS sre_documents (
            id        SERIAL PRIMARY KEY,
            content   TEXT NOT NULL,
            source    TEXT,
            embedding vector({EMBEDDING_DIM}),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # HNSW index for fast approximate nearest neighbor search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS sre_docs_embedding_idx
        ON sre_documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database setup complete ✅")

# ── Embedding Model ────────────────────────────────────────────────────────────

class EmbeddingModel:
    """Singleton wrapper for sentence-transformers model."""
    _instance = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            cls._instance = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded ✅")
        return cls._instance

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts — returns (N, 384) float32 array."""
    model = EmbeddingModel.get()
    return model.encode(texts, batch_size=64, show_progress_bar=False)

# ── Document Indexing ──────────────────────────────────────────────────────────

def load_and_chunk_documents(path: Path, max_docs: int = 2000) -> list[dict]:
    """
    Load training JSONL and extract text chunks for indexing.
    Extracts the answer portion from Mistral instruct format
    since answers contain the actual SRE knowledge.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " "],
    )

    chunks = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            try:
                rec  = json.loads(line.strip())
                text = rec.get("text", "")

                # Extract answer from Mistral instruct format
                # Format: <s>[INST] ... [/INST] ANSWER </s>
                if "[/INST]" in text:
                    answer = text.split("[/INST]")[-1].replace("</s>", "").strip()
                    question_part = text.split("[/INST]")[0]
                    # Extract question
                    question = question_part.split("\n\n")[-1].replace("[/INST]", "").strip()
                    content = f"Q: {question}\n\nA: {answer}"
                else:
                    content = text

                # Split into chunks
                doc_chunks = splitter.split_text(content)
                for chunk in doc_chunks:
                    if len(chunk) > 100:  # skip very short chunks
                        chunks.append({
                            "content": chunk,
                            "source":  f"train_{i}",
                        })

            except Exception:
                continue

    logger.info(f"Loaded {len(chunks)} chunks from {max_docs} documents")
    return chunks

def index_documents(chunks: list[dict], batch_size: int = 256):
    """
    Embed and insert document chunks into pgvector.
    Uses batch inserts for efficiency.
    """
    conn = get_vector_connection()
    cur  = conn.cursor()

    # Check if already indexed
    cur.execute("SELECT COUNT(*) FROM sre_documents;")
    count = cur.fetchone()[0]
    if count > 0:
        logger.info(f"Already indexed {count} documents — skipping")
        cur.close()
        conn.close()
        return count

    logger.info(f"Indexing {len(chunks)} chunks...")
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch   = chunks[i:i + batch_size]
        texts   = [c["content"] for c in batch]
        sources = [c["source"]  for c in batch]

        # Embed batch
        embeddings = embed(texts)

        # Insert into pgvector
        execute_values(
            cur,
            """
            INSERT INTO sre_documents (content, source, embedding)
            VALUES %s
            """,
            [
                (texts[j], sources[j], embeddings[j].tolist())
                for j in range(len(batch))
            ]
        )
        conn.commit()
        total += len(batch)
        logger.info(f"  Indexed {total}/{len(chunks)} chunks")

    cur.close()
    conn.close()
    logger.info(f"Indexing complete — {total} chunks in pgvector ✅")
    return total

# ── BM25 Index ─────────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index built from pgvector documents.
    Rebuilt on startup — fast enough for <10k docs.
    """
    _instance    = None
    _doc_ids     = None
    _doc_contents = None

    @classmethod
    def build(cls):
        """Load all documents from DB and build BM25 index."""
        if cls._instance is not None:
            return

        logger.info("Building BM25 index...")
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT id, content FROM sre_documents ORDER BY id;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        cls._doc_ids      = [r[0] for r in rows]
        cls._doc_contents = [r[1] for r in rows]

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in cls._doc_contents]
        cls._instance = BM25Okapi(tokenized)
        logger.info(f"BM25 index built — {len(rows)} documents ✅")

    @classmethod
    def search(cls, query: str, top_k: int = TOP_K_SPARSE) -> list[dict]:
        """Return top-k BM25 results for query."""
        if cls._instance is None:
            cls.build()

        tokens = query.lower().split()
        scores = cls._instance.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id":      cls._doc_ids[i],
                "content": cls._doc_contents[i],
                "score":   float(scores[i]),
                "source":  "bm25",
            }
            for i in top_indices
            if scores[i] > 0
        ]

# ── Dense Retrieval ────────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = TOP_K_DENSE) -> list[dict]:
    """pgvector cosine similarity search."""
    query_embedding = embed([query])[0]

    conn = get_vector_connection()
    cur  = conn.cursor()
    cur.execute(
        """
        SELECT id, content, 1 - (embedding <=> %s::vector) AS score
        FROM sre_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding.tolist(), query_embedding.tolist(), top_k)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {"id": r[0], "content": r[1], "score": float(r[2]), "source": "pgvector"}
        for r in rows
    ]

# ── Hybrid Fusion ──────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight:  float = 0.6,
    sparse_weight: float = 0.4,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) combines dense + sparse rankings.
    RRF score = dense_weight/(k + rank_dense) + sparse_weight/(k + rank_sparse)
    k=60 is standard — reduces impact of very high ranks.
    """
    scores = {}
    contents = {}

    # Score dense results
    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        scores[doc_id]   = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
        contents[doc_id] = doc["content"]

    # Score sparse results
    for rank, doc in enumerate(sparse_results):
        doc_id = doc["id"]
        scores[doc_id]   = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
        contents[doc_id] = doc.get("content", contents.get(doc_id, ""))

    # Sort by fused score
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [
        {
            "id":      doc_id,
            "content": contents[doc_id],
            "score":   round(scores[doc_id], 6),
            "source":  "hybrid_rrf",
        }
        for doc_id in sorted_ids[:TOP_K_FINAL]
    ]

# ── Main Retriever ─────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_FINAL) -> list[str]:
    """
    Main retrieval function — hybrid pgvector + BM25.
    Returns list of document content strings for LLM context.
    """
    dense_results  = dense_search(query, top_k=TOP_K_DENSE)
    sparse_results = BM25Index.search(query, top_k=TOP_K_SPARSE)
    fused_results  = reciprocal_rank_fusion(dense_results, sparse_results)

    logger.info(
        f"Retrieved {len(fused_results)} docs — "
        f"dense: {len(dense_results)}, sparse: {len(sparse_results)}"
    )
    return [doc["content"] for doc in fused_results[:top_k]]

# ── Full RAG Query ─────────────────────────────────────────────────────────────

def rag_query(question: str) -> dict:
    """
    End-to-end RAG query:
    1. Retrieve relevant docs (hybrid)
    2. Route to LLM with context
    Returns full response with docs and metadata.
    """
    import sys
    sys.path.append(str(Path(__file__).parent))
    from llm_router import route_llm

    # Step 1 — retrieve
    retrieved_docs = retrieve(question)

    # Step 2 — generate
    result = route_llm(
        question       = question,
        retrieved_docs = retrieved_docs,
        force_backend  = "sre_ollama",
    )

    return {
        "question":      question,
        "answer":        result["answer"],
        "backend":       result["backend"],
        "latency_ms":    result["latency_ms"],
        "retrieved_docs": retrieved_docs,
        "num_docs":      len(retrieved_docs),
    }

# ── Setup Script ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Setting up Local RAG Pipeline ===\n")

    print("[1/3] Setting up database...")
    setup_database()

    print("\n[2/3] Loading and chunking documents...")
    chunks = load_and_chunk_documents(DATASET_PATH, max_docs=2000)

    print("\n[3/3] Indexing into pgvector...")
    total = index_documents(chunks)

    print("\n[4/4] Building BM25 index...")
    BM25Index.build()

    print(f"\n✅ RAG pipeline ready — {total} chunks indexed")
    print("Run queries with: from rag_pipeline import rag_query")