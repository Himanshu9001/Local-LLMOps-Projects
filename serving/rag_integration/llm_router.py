"""
Phase 7 — LLM Router for RAG Pipeline Integration
Plugs the local SRE Ollama assistant into an existing RAG pipeline
as a primary backend with Azure OpenAI fallback.

Architecture:
    RAG Query
        ↓
    Hybrid Retrieval (pgvector + BM25)
        ↓
    Cross-Encoder Reranker
        ↓
    LLM Router (this file)
        ├── SRE Ollama (local, primary for SRE/DevOps queries)
        └── Azure OpenAI GPT-4o (fallback)
"""

import os
import time
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SRE_API_URL     = os.environ.get("SRE_API_URL",     "http://localhost:8000")
SRE_ENABLED     = os.environ.get("SRE_ENABLED",     "true").lower() == "true"
SRE_TIMEOUT_SEC = int(os.environ.get("SRE_TIMEOUT", "30"))

AZURE_OPENAI_KEY      = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_MODEL    = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o")

# SRE domain keywords — route these to local model
SRE_DOMAIN_KEYWORDS = [
    "kubernetes", "k8s", "pod", "deployment", "ingress", "helm",
    "terraform", "kubectl", "docker", "container", "namespace",
    "crashloopbackoff", "oomkilled", "evicted", "pending",
    "prometheus", "grafana", "alertmanager", "pagerduty",
    "github actions", "argocd", "ci/cd", "pipeline",
    "aws", "azure", "gcp", "eks", "aks", "gke",
    "incident", "postmortem", "sre", "devops", "oncall",
    "error", "debug", "troubleshoot", "fix", "root cause",
]

# ── Domain Detection ───────────────────────────────────────────────────────────
def is_sre_query(question: str) -> bool:
    """
    Detect if a query is SRE/DevOps domain.
    Routes to local fine-tuned model if true.
    Simple keyword matching — can be replaced with a classifier.
    """
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in SRE_DOMAIN_KEYWORDS)

# ── SRE Ollama Backend ─────────────────────────────────────────────────────────
def call_sre_assistant(
    question:       str,
    context:        str = "",
    retrieved_docs: list[str] = [],
) -> tuple[str, str]:
    """
    Call the local SRE assistant FastAPI service.
    Returns (answer, source) tuple.
    source = "sre_ollama" | "fallback_needed"
    """
    if not SRE_ENABLED:
        return "", "fallback_needed"

    # Combine retrieved docs into context
    full_context = context
    if retrieved_docs:
        docs_text   = "\n\n".join(retrieved_docs[:3])  # top 3 docs
        full_context = f"Retrieved Documents:\n{docs_text}\n\n{context}".strip()

    try:
        with httpx.Client(timeout=SRE_TIMEOUT_SEC) as client:
            response = client.post(
                f"{SRE_API_URL}/query",
                json={
                    "question": question,
                    "context":  full_context,
                }
            )
            response.raise_for_status()
            data   = response.json()
            answer = data.get("answer", "").strip()

            if not answer:
                logger.warning("SRE assistant returned empty answer — falling back")
                return "", "fallback_needed"

            logger.info(f"SRE assistant responded in {data.get('latency_ms', 0):.0f}ms")
            return answer, "sre_ollama"

    except httpx.ConnectError:
        logger.warning("SRE assistant unreachable — falling back to Azure OpenAI")
        return "", "fallback_needed"
    except httpx.TimeoutException:
        logger.warning(f"SRE assistant timed out after {SRE_TIMEOUT_SEC}s — falling back")
        return "", "fallback_needed"
    except Exception as e:
        logger.error(f"SRE assistant error: {e} — falling back")
        return "", "fallback_needed"

# ── Azure OpenAI Fallback ──────────────────────────────────────────────────────
def call_azure_openai(
    question:       str,
    context:        str = "",
    retrieved_docs: list[str] = [],
) -> tuple[str, str]:
    """
    Fallback to Azure OpenAI GPT-4o.
    Returns (answer, source) tuple.
    """
    if not AZURE_OPENAI_KEY:
        return "Azure OpenAI not configured.", "error"

    full_context = context
    if retrieved_docs:
        docs_text    = "\n\n".join(retrieved_docs[:3])
        full_context = f"Retrieved Documents:\n{docs_text}\n\n{context}".strip()

    system_prompt = (
        "You are an expert SRE and DevOps engineer. "
        "Answer based on the provided context. Be concise and actionable."
    )
    user_message = question
    if full_context:
        user_message = f"Context:\n{full_context}\n\nQuestion: {question}"

    try:
        import openai
        client = openai.AzureOpenAI(
            api_key         = AZURE_OPENAI_KEY,
            api_version     = "2024-02-01",
            azure_endpoint  = AZURE_OPENAI_ENDPOINT,
        )
        response = client.chat.completions.create(
            model    = AZURE_OPENAI_MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 500,
            temperature = 0.1,
        )
        return response.choices[0].message.content.strip(), "azure_openai"

    except Exception as e:
        logger.error(f"Azure OpenAI error: {e}")
        return f"Error: {str(e)}", "error"

# ── Main Router ────────────────────────────────────────────────────────────────
def route_llm(
    question:       str,
    context:        str = "",
    retrieved_docs: list[str] = [],
    force_backend:  Optional[str] = None,
) -> dict:
    """
    Main LLM router — selects backend based on query domain.

    Args:
        question:       User question
        context:        Optional additional context
        retrieved_docs: Docs from RAG retrieval step
        force_backend:  Override routing — "sre_ollama" | "azure_openai"

    Returns:
        {
            "answer":      str,
            "backend":     str,   # which backend answered
            "domain":      str,   # "sre" | "general"
            "latency_ms":  float,
        }
    """
    start  = time.time()
    domain = "sre" if is_sre_query(question) else "general"

    # Determine backend
    use_sre = (
        force_backend == "sre_ollama" or
        (force_backend is None and domain == "sre" and SRE_ENABLED)
    )

    if use_sre:
        logger.info(f"Routing to SRE Ollama — domain={domain}")
        answer, backend = call_sre_assistant(question, context, retrieved_docs)

        # Fallback if SRE assistant failed
        if backend == "fallback_needed":
            logger.info("SRE failed — falling back to Azure OpenAI")
            answer, backend = call_azure_openai(question, context, retrieved_docs)
    else:
        logger.info(f"Routing to Azure OpenAI — domain={domain}")
        answer, backend = call_azure_openai(question, context, retrieved_docs)

    latency = (time.time() - start) * 1000

    return {
        "answer":     answer,
        "backend":    backend,
        "domain":     domain,
        "latency_ms": round(latency, 2),
    }