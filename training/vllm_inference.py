"""
vLLM Inference Script — SRE/DevOps Mistral-7B
Platform: NVIDIA GPU (T4/V100/A100) — Kaggle, RunPod, Lambda Labs, AWS
Usage:
    python training/vllm_inference.py
    python training/vllm_inference.py --model Himanshu0910/sre-devops-mistral-7b
    python training/vllm_inference.py --serve          # start OpenAI-compatible server
    python training/vllm_inference.py --benchmark      # run throughput benchmark

Requirements:
    pip install vllm openai
    NVIDIA GPU with CUDA support required
"""

import os
import time
import argparse

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="vLLM inference for SRE assistant")
parser.add_argument("--model",       type=str, default="Himanshu0910/sre-devops-mistral-7b")
parser.add_argument("--dtype",       type=str, default="float16", choices=["float16", "bfloat16", "auto"])
parser.add_argument("--max_len",     type=int, default=2048)
parser.add_argument("--gpu_util",    type=float, default=0.85)
parser.add_argument("--tp",          type=int, default=1, help="Tensor parallel size (GPUs)")
parser.add_argument("--serve",       action="store_true", help="Start OpenAI-compatible server")
parser.add_argument("--port",        type=int, default=8000)
parser.add_argument("--benchmark",   action="store_true", help="Run throughput benchmark")
parser.add_argument("--max_tokens",  type=int, default=300)
parser.add_argument("--temperature", type=float, default=0.1)
args = parser.parse_args()

# ── Test prompts ───────────────────────────────────────────────────────────────
SRE_PROMPTS = [
    "<s>[INST] You are an expert SRE engineer. What does Exit Code 137 mean in Kubernetes and how do I fix it? [/INST]",
    "<s>[INST] You are an expert SRE engineer. How do I safely unlock a Terraform state lock? [/INST]",
    "<s>[INST] You are an expert SRE engineer. My ArgoCD app is OutOfSync. What are the debugging steps? [/INST]",
    "<s>[INST] You are an expert SRE engineer. My Kubernetes node is in NotReady state. How do I diagnose it? [/INST]",
    "<s>[INST] You are an expert SRE engineer. What is the difference between a Kubernetes Deployment and StatefulSet? [/INST]",
    "<s>[INST] You are an expert SRE engineer. My pod is stuck in Pending state. What are the possible causes? [/INST]",
    "<s>[INST] You are an expert SRE engineer. How do I debug high memory usage in a Kubernetes pod? [/INST]",
    "<s>[INST] You are an expert SRE engineer. My GitHub Actions pipeline is failing on docker build. How do I debug? [/INST]",
]

# ── Environment check ──────────────────────────────────────────────────────────
def check_environment():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected.\n"
            "vLLM requires an NVIDIA GPU.\n"
            "For Apple Silicon, use Ollama instead: ollama run sre-assistant"
        )
    print(f"CUDA available: True")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {name} | VRAM: {vram:.1f}GB")

# ── Load vLLM model ────────────────────────────────────────────────────────────
def load_model():
    from vllm import LLM, SamplingParams

    print(f"\nLoading model: {args.model}")
    print(f"dtype: {args.dtype} | max_len: {args.max_len} | gpu_util: {args.gpu_util}")
    print(f"tensor_parallel_size: {args.tp}")

    llm = LLM(
        model                  = args.model,
        dtype                  = args.dtype,
        max_model_len          = args.max_len,
        gpu_memory_utilization = args.gpu_util,
        tensor_parallel_size   = args.tp,
    )
    print("Model loaded ✅\n")
    return llm

# ── Batch inference ────────────────────────────────────────────────────────────
def run_inference(llm):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature        = args.temperature,
        max_tokens         = args.max_tokens,
        repetition_penalty = 1.3,
    )

    print("=== SRE Domain Inference Test ===\n")
    outputs = llm.generate(SRE_PROMPTS[:5], sampling_params)

    for i, output in enumerate(outputs):
        question = output.prompt.split("[INST]")[1].split("[/INST]")[0].strip()
        answer   = output.outputs[0].text.strip()
        tokens   = len(output.outputs[0].token_ids)

        print(f"Q{i+1}: {question[:80]}")
        print(f"A:  {answer[:300]}")
        print(f"    [{tokens} tokens]")
        print()

# ── Throughput benchmark ───────────────────────────────────────────────────────
def run_benchmark(llm):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature        = args.temperature,
        max_tokens         = args.max_tokens,
        repetition_penalty = 1.3,
    )

    print("=== vLLM Throughput Benchmark ===\n")
    print(f"Model:       {args.model}")
    print(f"Batch size:  {len(SRE_PROMPTS)}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"GPUs:        {args.tp}\n")

    # Warmup
    print("Warmup run...")
    llm.generate(SRE_PROMPTS[:2], sampling_params)

    # Benchmark
    print("Benchmark run...")
    start   = time.time()
    outputs = llm.generate(SRE_PROMPTS, sampling_params)
    elapsed = time.time() - start

    total_tokens    = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_input_tok = sum(len(o.prompt_token_ids) for o in outputs)
    throughput      = total_tokens / elapsed
    avg_latency     = elapsed / len(SRE_PROMPTS) * 1000

    print(f"\nResults:")
    print(f"  Requests:        {len(SRE_PROMPTS)}")
    print(f"  Input tokens:    {total_input_tok}")
    print(f"  Output tokens:   {total_tokens}")
    print(f"  Total time:      {elapsed:.2f}s")
    print(f"  Throughput:      {throughput:.1f} tok/s")
    print(f"  Avg latency:     {avg_latency:.0f}ms per request")
    print()

    # Per-request breakdown
    print("Per-request breakdown:")
    for i, output in enumerate(outputs):
        tokens = len(output.outputs[0].token_ids)
        print(f"  Q{i+1}: {tokens} tokens")

    print(f"\nComparison:")
    print(f"  vLLM batch ({args.tp} GPU):  {throughput:.1f} tok/s")
    print(f"  Ollama sequential (M1):  ~20-30 tok/s")
    print(f"  Speedup: ~{throughput/25:.1f}x vs Ollama")

# ── OpenAI-compatible server ───────────────────────────────────────────────────
def start_server():
    import subprocess
    import requests

    print(f"Starting vLLM OpenAI-compatible server on port {args.port}...")
    print(f"Model: {args.model}")
    print(f"Compatible with: OpenAI Python client, curl, FastAPI\n")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  args.model,
        "--dtype",                  args.dtype,
        "--max-model-len",          str(args.max_len),
        "--host",                   "0.0.0.0",
        "--port",                   str(args.port),
        "--gpu-memory-utilization", str(args.gpu_util),
        "--tensor-parallel-size",   str(args.tp),
    ]

    proc = subprocess.Popen(cmd)

    # Wait for server to be ready
    print("Waiting for server to start...")
    for i in range(60):
        try:
            resp = requests.get(f"http://localhost:{args.port}/health", timeout=2)
            if resp.status_code == 200:
                print(f"\nvLLM server ready ✅")
                print(f"API URL:  http://localhost:{args.port}")
                print(f"Docs URL: http://localhost:{args.port}/docs")
                break
        except Exception:
            pass
        time.sleep(5)
        print(f"  {i*5}s elapsed...")

    print("\nExample usage:")
    print(f"""
# curl
curl -X POST http://localhost:{args.port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{args.model}", "messages": [{{"role": "user", "content": "What is Exit Code 137?"}}]}}'

# Python OpenAI client
import openai
client = openai.OpenAI(base_url="http://localhost:{args.port}/v1", api_key="dummy")
response = client.chat.completions.create(
    model="{args.model}",
    messages=[{{"role": "user", "content": "What is Exit Code 137?"}}],
    max_tokens=300,
)
print(response.choices[0].message.content)

# Switch FastAPI backend from Ollama to vLLM — change ONE line in serving/api/main.py:
# OLLAMA_HOST = "http://localhost:{args.port}/v1"
    """)

    # Keep server running
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        proc.terminate()

# ── Interactive mode ───────────────────────────────────────────────────────────
def interactive_mode(llm):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature        = args.temperature,
        max_tokens         = args.max_tokens,
        repetition_penalty = 1.3,
    )

    print("=== Interactive SRE Assistant ===")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        prompt  = f"<s>[INST] You are an expert SRE engineer. {question} [/INST]"
        outputs = llm.generate([prompt], sampling_params)
        answer  = outputs[0].outputs[0].text.strip()
        tokens  = len(outputs[0].outputs[0].token_ids)

        print(f"\nAssistant: {answer}")
        print(f"[{tokens} tokens]\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SRE/DevOps AI Assistant — vLLM Inference")
    print("=" * 60)

    # Server mode doesn't need pre-loading
    if args.serve:
        start_server()
        return

    # All other modes need the model loaded
    check_environment()
    llm = load_model()

    if args.benchmark:
        run_benchmark(llm)
    else:
        run_inference(llm)
        print("\n--- Interactive mode ---")
        interactive_mode(llm)

if __name__ == "__main__":
    main()