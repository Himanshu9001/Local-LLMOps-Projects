"""
Phase 6 — FastAPI wrapper for SRE/DevOps AI Assistant
Wraps Ollama sre-assistant model in a production REST API
Run: uvicorn serving.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SRE/DevOps AI Assistant API",
    description = "REST API for fine-tuned Mistral-7B SRE assistant running on Ollama",
    version     = "1.0.0",
)

# ── CORS — allow Gradio + RAG pipeline to call this API ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "sre-assistant"
DEFAULT_OPTIONS = {
    "temperature":    0.1,
    "top_p":          0.85,
    "repeat_penalty": 1.5,
    "num_predict":    400,
}

# ── Schemas ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    context:   Optional[str] = ""
    stream:    Optional[bool] = False

class QueryResponse(BaseModel):
    request_id:    str
    answer:        str
    model:         str
    latency_ms:    float

class HealthResponse(BaseModel):
    status:  str
    model:   str
    ollama:  str

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    stream:   Optional[bool] = False

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_prompt(question: str, context: str) -> str:
    """Combine question and context into a single user message."""
    if context and context.strip():
        return f"Context/Logs:\n{context.strip()}\n\nQuestion: {question.strip()}"
    return question.strip()

def call_ollama(messages: list[dict]) -> str:
    """Call Ollama and return response text."""
    try:
        response = ollama.chat(
            model    = MODEL_NAME,
            messages = messages,
            options  = DEFAULT_OPTIONS,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(
            status_code = 503,
            detail      = f"Ollama unavailable: {str(e)}. Run: ollama serve"
        )

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — verifies Ollama is reachable and model is loaded."""
    try:
        models = ollama.list()
        model_names = [m["model"] for m in models.get("models", [])]
        ollama_status = "ok"
        model_status  = MODEL_NAME if any(MODEL_NAME in m for m in model_names) else "not loaded"
    except Exception as e:
        ollama_status = f"unreachable: {str(e)}"
        model_status  = "unknown"

    return HealthResponse(
        status = "ok",
        model  = model_status,
        ollama = ollama_status,
    )

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Primary endpoint — single question + optional context.
    Used by Gradio dashboard and RAG pipeline.
    """
    request_id = str(uuid.uuid4())[:8]
    start      = time.time()

    prompt   = build_prompt(req.question, req.context or "")
    messages = [{"role": "user", "content": prompt}]
    answer   = call_ollama(messages)

    latency = (time.time() - start) * 1000

    return QueryResponse(
        request_id = request_id,
        answer     = answer,
        model      = MODEL_NAME,
        latency_ms = round(latency, 2),
    )

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Multi-turn chat endpoint — accepts full message history.
    Used for conversational debugging sessions.
    """
    request_id = str(uuid.uuid4())[:8]
    start      = time.time()

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    answer   = call_ollama(messages)
    latency  = (time.time() - start) * 1000

    return {
        "request_id": request_id,
        "answer":     answer,
        "model":      MODEL_NAME,
        "latency_ms": round(latency, 2),
    }

@app.get("/models")
def list_models():
    """List all models available in Ollama."""
    try:
        return ollama.list()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/")
def root():
    return {
        "service": "SRE/DevOps AI Assistant API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": ["/query", "/chat", "/models", "/health"],
    }