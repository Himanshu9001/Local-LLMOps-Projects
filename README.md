# SRE/DevOps AI Assistant — Local LLMOps Pipeline

A production-grade, end-to-end LLMOps pipeline that fine-tunes **Mistral-7B** on SRE/DevOps domain data and serves it locally with full RAG retrieval — zero cloud dependency, zero API cost.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Mistral-7B](https://img.shields.io/badge/Model-Mistral--7B--Instruct--v0.3-orange)](https://huggingface.co/Himanshu0910/sre-devops-mistral-7b)
[![MLflow](https://img.shields.io/badge/MLflow-3.12-blue)](https://mlflow.org)
[![Ollama](https://img.shields.io/badge/Serving-Ollama-black)](https://ollama.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com)
[![Kubernetes](https://img.shields.io/badge/Deploy-Kubernetes-326CE5)](https://kubernetes.io)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection (Phase 1)                     │
│  Kubernetes Docs · AWS Docs · Azure Docs · Stack Overflow · GitHub│
│                    6,074 raw records                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Dataset Preparation (Phase 2)                   │
│         Cleaning · Q&A Generation · Mistral Instruct Format      │
│                    7,302 training examples                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   QLoRA Fine-Tuning (Phase 3)                    │
│        Mistral-7B-Instruct-v0.3 · r=16 · 700 steps · T4 GPU     │
│              train_loss: 0.85 · val_loss: 1.05                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Local Serving Stack (Phase 4-6)                 │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Gradio    │    │   FastAPI   │    │   Ollama + llama.cpp │  │
│  │  :7860      │───▶│   :8000     │───▶│       :11434        │  │
│  │  Dashboard  │    │  REST API   │    │  Q4_K_M (4.17GB)    │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Local RAG Pipeline (Phase 7B)                  │
│                                                                  │
│  User Query                                                      │
│      ↓                                                           │
│  pgvector (dense)  +  BM25 (sparse)  →  RRF Fusion              │
│      ↓                                                           │
│  Top 5 retrieved docs                                            │
│      ↓                                                           │
│  LLM Router                                                      │
│  ├── SRE Ollama (local, fine-tuned) ← primary for SRE queries   │
│  └── Azure OpenAI GPT-4o            ← fallback                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              Kubernetes Deployment + MLflow Registry             │
│     minikube · Ollama pod · FastAPI pod · vLLM config            │
│     MLflow model registry · loss curves · 30 params logged      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Web scraping — 5 sources, 6,074 records | ✅ |
| 2 | Dataset prep — 7,302 Mistral instruct format examples | ✅ |
| 3 | QLoRA fine-tuning — Mistral-7B, 700 steps on T4 GPU | ✅ |
| 4 | Local serving — Ollama Q4_K_M on Apple M1 Metal | ✅ |
| 5 | Gradio dashboard — interactive chat UI | ✅ |
| 6 | FastAPI wrapper — REST API with /query /chat /health | ✅ |
| 7 | RAG integration — LLM router with Azure OpenAI fallback | ✅ |
| 7B | Local RAG — pgvector + BM25 hybrid, 7,519 chunks indexed | ✅ |
| MLflow | Model registry — v1, loss curves, 30 params logged | ✅ |
| K8s | Kubernetes deployment — Ollama + FastAPI on minikube | ✅ |

---

## Dataset

| Source | Records | Format |
|--------|---------|--------|
| Kubernetes docs | 1,061 | Raw text → Q&A generated |
| Azure docs (AKS, Terraform, Monitor) | 2,206 | Raw text → Q&A generated |
| AWS docs (EKS, CloudWatch, IAM) | 1,988 | Raw text → Q&A generated |
| Stack Overflow (DevOps tags) | 1,813 | Already Q&A format |
| GitHub postmortems (danluu/post-mortems) | 234 | Incident summaries → RCA Q&A |
| **Total** | **7,302** | Mistral instruct format |

---

## Training Configuration

```yaml
base_model:          mistralai/Mistral-7B-Instruct-v0.3
method:              QLoRA
lora_r:              16
lora_alpha:          32
lora_dropout:        0.05
target_modules:      q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
learning_rate:       2e-4
batch_size:          1
gradient_accumulation: 16
effective_batch:     16
epochs:              3
max_seq_length:      1024
optimizer:           adamw_8bit
scheduler:           cosine
quantization:        4bit NF4
steps_completed:     700 / 1230 (56.9%)
final_train_loss:    0.8544
final_val_loss:      1.0549
```

### Loss Curve

| Step | Train Loss | Val Loss |
|------|-----------|---------|
| 50 | 1.1662 | 1.1853 |
| 150 | 1.1112 | 1.0962 |
| 300 | 1.0793 | 1.0793 |
| 450 | 0.8368 | 1.0735 |
| 600 | 0.8685 | 1.0588 |
| 700 | 0.8544 | 1.0549 |

---

## Repository Structure

```
Local-LLMOps-Projects/
├── data/
│   ├── scrapers/
│   │   ├── k8s_scraper.py          # Kubernetes docs scraper
│   │   ├── azure_scraper.py        # Azure docs BFS scraper
│   │   ├── aws_scraper.py          # AWS sitemap-driven scraper
│   │   ├── github_scraper.py       # GitHub postmortem scraper
│   │   ├── stackoverflow_scraper.py # Stack Exchange API scraper
│   │   └── run_all.py              # Master runner (resumable)
│   ├── prepare_dataset.py          # Raw → Mistral instruct JSONL
│   └── processed/
│       └── dataset_stats.json
│
├── training/
│   └── colab_train.ipynb           # Colab QLoRA training notebook
│
├── serving/
│   ├── ollama/
│   │   └── Modelfile               # Ollama model definition
│   ├── ui/
│   │   └── app.py                  # Gradio dashboard
│   ├── api/
│   │   ├── main.py                 # FastAPI service
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── rag_integration/
│   │   ├── rag_pipeline.py         # pgvector + BM25 hybrid RAG
│   │   ├── llm_router.py           # LLM router with fallback
│   │   └── test_router.py          # End-to-end test
│   └── mlflow_registry/
│       └── register_model.py       # MLflow model registration
│
├── k8s/
│   ├── ollama-deployment.yaml      # Ollama K8s deployment
│   ├── fastapi-deployment.yaml     # FastAPI K8s deployment
│   └── vllm-deployment.yaml        # vLLM production config (NVIDIA GPU)
│
├── TROUBLESHOOTING.md              # 15 real issues + fixes + lessons
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
# Required
brew install ollama colima minikube
python3 -m venv .venv-local-llmops
source .venv-local-llmops/bin/activate
pip install -r requirements.txt
```

### 1. Start services

```bash
# Start Ollama
ollama serve &

# Start pgvector (for RAG)
colima start
docker run -d --name sre-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=sre_rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Create Ollama model

```bash
ollama create sre-assistant -f serving/ollama/Modelfile
ollama run sre-assistant "My pod is in CrashLoopBackOff. Root cause?"
```

### 3. Start FastAPI

```bash
uvicorn serving.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Start Gradio dashboard

```bash
python serving/ui/app.py
# Open: http://localhost:7860
```

### 5. Index RAG corpus

```bash
cd serving/rag_integration
python rag_pipeline.py
# Indexes 7,519 chunks into pgvector + BM25
```

### 6. Run full RAG query

```python
from rag_pipeline import rag_query
result = rag_query("What does Exit Code 137 mean in Kubernetes?")
print(result["answer"])
```

---

## API Reference

### FastAPI endpoints (`localhost:8000`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check + model status |
| `/query` | POST | Single question + optional context |
| `/chat` | POST | Multi-turn conversation |
| `/models` | GET | List available Ollama models |
| `/docs` | GET | Auto-generated Swagger UI |

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "My ArgoCD app is OutOfSync. How do I debug?",
       "context": "kubectl get app myapp -n argocd shows degraded"}'
```

---

## Kubernetes Deployment

### Local (minikube)

```bash
# Start cluster
colima start --cpu 2 --memory 6 --disk 60
minikube start --memory 4096 --cpus 2 --driver docker

# Deploy
kubectl apply -f k8s/ollama-deployment.yaml
kubectl apply -f k8s/fastapi-deployment.yaml

# Access
minikube service sre-api-service --url
```

### Production (NVIDIA GPU — vLLM)

```bash
# Requires NVIDIA GPU node with device plugin installed
kubectl apply -f k8s/vllm-deployment.yaml

# Create HF token secret first
kubectl create secret generic hf-secret \
  --from-literal=token=your_hf_token
```

---

## MLflow Model Registry

```bash
# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 --port 5000

# Register model
python serving/mlflow_registry/register_model.py

# View at: http://localhost:5000
```

**Registered model:** `sre-devops-mistral-7b` v1
- 30 parameters logged
- Full loss curve (steps 50-700)
- Model card artifact
- Dataset statistics

---

## Model Card

**Model:** [Himanshu0910/sre-devops-mistral-7b](https://huggingface.co/Himanshu0910/sre-devops-mistral-7b)

**Base model:** mistralai/Mistral-7B-Instruct-v0.3

**Intended use:** SRE and DevOps troubleshooting assistance — Kubernetes debugging, infrastructure incident analysis, CI/CD pipeline issues.

**Known limitations:**
- Training completed 700/1230 steps (56.9%) — model shows mild hallucination patterns from Stack Overflow training data
- `repeat_penalty >= 1.3` required to prevent output loops
- Best results when used with RAG retrieval for document grounding
- Completing full training (1230 steps) would significantly improve quality

**Training data sources:** Kubernetes docs, AWS docs, Azure docs, Stack Overflow DevOps tags, GitHub incident postmortems — all publicly available, CC BY-SA 4.0 or Apache 2.0 licensed.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base model | Mistral-7B-Instruct-v0.3 |
| Fine-tuning | QLoRA via Unsloth + HuggingFace TRL |
| Local inference | Ollama + llama.cpp (Metal GPU) |
| Quantization | GGUF Q4_K_M |
| Vector DB | PostgreSQL + pgvector (HNSW index) |
| Sparse retrieval | BM25Okapi (rank-bm25) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| API | FastAPI + Uvicorn |
| UI | Gradio 6.x |
| Model registry | MLflow 3.12 |
| Container orchestration | Kubernetes (minikube) |
| Production serving | vLLM (NVIDIA GPU config) |
| Experiment tracking | MLflow |

---

## Data Sources & Licensing

| Source | License | Usage |
|--------|---------|-------|
| kubernetes.io/docs | CC BY 4.0 | Training data |
| learn.microsoft.com/azure | Microsoft Permissive | Training data |
| docs.aws.amazon.com | CC BY-SA 4.0 | Training data |
| stackoverflow.com (DevOps) | CC BY-SA 4.0 | Training data |
| github.com/danluu/post-mortems | MIT | Training data |

This model is intended for research and personal portfolio use. Commercial use requires reviewing individual source licenses.

---

## Author

**Himanshu Singh** — Cloud DevOps & AI Engineer  
GitHub: [@Himanshu9001](https://github.com/Himanshu9001)  
HuggingFace: [@Himanshu0910](https://huggingface.co/Himanshu0910)