"""
Test the LLM router end-to-end.
Run: python serving/rag_integration/test_router.py
Make sure FastAPI is running: uvicorn serving.api.main:app --port 8000
"""

from llm_router import route_llm, is_sre_query

def test_domain_detection():
    """Verify SRE domain detection."""
    sre_queries = [
        "My Kubernetes pod is in CrashLoopBackOff",
        "Terraform state lock issue",
        "ArgoCD sync failed",
        "How do I debug an OOMKilled container?",
    ]
    general_queries = [
        "What is machine learning?",
        "How do I write a Python function?",
        "What is the capital of France?",
    ]

    print("=== Domain Detection Test ===")
    for q in sre_queries:
        result = is_sre_query(q)
        status = "✅" if result else "❌"
        print(f"  {status} SRE: {q[:50]}")

    for q in general_queries:
        result = is_sre_query(q)
        status = "✅" if not result else "❌"
        print(f"  {status} General: {q[:50]}")

def test_routing():
    """Test full routing with mock retrieved docs."""
    print("\n=== Routing Test ===")

    # Simulate RAG retrieved docs
    mock_docs = [
        "Exit Code 137 in Kubernetes means the container was killed by the OOMKiller.",
        "To fix OOMKilled: increase memory limits in your deployment manifest.",
    ]

    result = route_llm(
        question       = "What does Exit Code 137 mean?",
        retrieved_docs = mock_docs,
    )

    print(f"  Backend:    {result['backend']}")
    print(f"  Domain:     {result['domain']}")
    print(f"  Latency:    {result['latency_ms']}ms")
    print(f"  Answer:     {result['answer'][:200]}")

if __name__ == "__main__":
    test_domain_detection()
    test_routing()