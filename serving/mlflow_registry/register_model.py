"""
Register SRE/DevOps fine-tuned Mistral-7B in MLflow Model Registry
Logs training params, metrics, artifacts, and model files
"""

import mlflow
import mlflow.pyfunc
import os
import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "sre-devops-llm-finetuning"
MODEL_NAME          = "sre-devops-mistral-7b"

GGUF_PATH     = "serving/ollama/sre-mistral-7b-q4.gguf"
MODELFILE     = "serving/ollama/Modelfile"
DATASET_STATS = "data/processed/dataset_stats.json"

# ── MLflow Custom Model Wrapper ────────────────────────────────────────────────
class SREAssistantModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper for the SRE assistant.
    Allows serving via MLflow's model server.
    Calls local Ollama endpoint for inference.
    """
    def load_context(self, context):
        import ollama
        self.client     = ollama
        self.model_name = "sre-assistant"

    def predict(self, context, model_input):
        import pandas as pd

        # Accept DataFrame with 'question' column
        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].tolist()
        else:
            questions = [model_input]

        results = []
        for question in questions:
            try:
                response = self.client.chat(
                    model    = self.model_name,
                    messages = [{"role": "user", "content": question}],
                    options  = {"temperature": 0.1, "num_predict": 400}
                )
                results.append(response["message"]["content"].strip())
            except Exception as e:
                results.append(f"Error: {str(e)}")

        return results

# ── Registration ───────────────────────────────────────────────────────────────
def register():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load dataset stats
    stats = {}
    if Path(DATASET_STATS).exists():
        with open(DATASET_STATS) as f:
            stats = json.load(f)

    with mlflow.start_run(run_name="mistral-7b-qlora-700steps") as run:

        # ── Log training parameters ────────────────────────────────────────────
        mlflow.log_params({
            "base_model":                "mistralai/Mistral-7B-Instruct-v0.3",
            "training_method":           "QLoRA",
            "lora_r":                    16,
            "lora_alpha":                32,
            "lora_dropout":              0.05,
            "lora_target_modules":       "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "learning_rate":             2e-4,
            "batch_size":                1,
            "gradient_accumulation":     16,
            "effective_batch_size":      16,
            "num_epochs":                3,
            "max_seq_length":            2048,
            "optimizer":                 "adamw_8bit",
            "lr_scheduler":              "cosine",
            "quantization":              "4bit_nf4",
            "training_steps_completed":  700,
            "training_steps_total":      1230,
            "training_completion_pct":   56.9,
        })

        # ── Log dataset metrics ────────────────────────────────────────────────
        mlflow.log_params({
            "dataset_total_examples":    stats.get("total", 7302),
            "dataset_stackoverflow":     stats.get("stackoverflow", 1813),
            "dataset_kubernetes":        stats.get("kubernetes", 1061),
            "dataset_azure":             stats.get("azure", 2206),
            "dataset_aws":               stats.get("aws", 1988),
            "dataset_postmortems":       stats.get("postmortems", 234),
        })

        # ── Log training metrics (from Colab loss table) ───────────────────────
        loss_curve = {
            50:  (1.1662, 1.1853),
            100: (1.0890, 1.1244),
            150: (1.1112, 1.0962),
            200: (1.0907, 1.1012),
            250: (1.0643, 1.0894),
            300: (1.0793, 1.0793),
            350: (0.9891, 1.0669),
            400: (1.0890, 1.0585),
            450: (0.8368, 1.0735),
            500: (0.8044, 1.0754),
            550: (0.8534, 1.0664),
            600: (0.8685, 1.0588),
            650: (0.8544, 1.0549),
            700: (0.8544, 1.0549),
        }

        for step, (train_loss, val_loss) in loss_curve.items():
            mlflow.log_metric("train_loss", train_loss, step=step)
            mlflow.log_metric("val_loss",   val_loss,   step=step)

        # ── Log serving config ─────────────────────────────────────────────────
        mlflow.log_params({
            "serving_framework":   "ollama",
            "quantization_format": "GGUF_Q4_K_M",
            "gguf_size_gb":        4.17,
            "embedding_dim":       4096,
            "context_length":      2048,
            "inference_backend":   "llama.cpp_metal",
        })

        # ── Log artifacts ──────────────────────────────────────────────────────
        if Path(MODELFILE).exists():
            mlflow.log_artifact(MODELFILE, artifact_path="config")

        if Path(DATASET_STATS).exists():
            mlflow.log_artifact(DATASET_STATS, artifact_path="data")

        # Log model card
        model_card = """# SRE/DevOps AI Assistant — Mistral-7B QLoRA

## Model Description
Fine-tuned Mistral-7B-Instruct-v0.3 on SRE/DevOps corpus using QLoRA.

## Training Data
- Kubernetes docs: 1,061 examples
- Azure docs: 2,206 examples
- AWS docs: 1,988 examples
- Stack Overflow (DevOps tags): 1,813 examples
- GitHub postmortems: 234 examples
- **Total: 7,302 examples**

## Training Config
- Method: QLoRA (r=16, alpha=32)
- Steps: 700/1230 (56.9% complete)
- Final train loss: 0.85
- Final val loss: 1.05

## Known Limitations
- Training incomplete (700/1230 steps)
- Tendency to hallucinate Stack Overflow context
- Best used with RAG retrieval for grounding

## Serving
- Format: GGUF Q4_K_M (4.17GB)
- Runtime: Ollama + llama.cpp Metal (Apple M1)
- API: FastAPI wrapper on port 8000
        """
        with open("/tmp/MODEL_CARD.md", "w") as f:
            f.write(model_card)
        mlflow.log_artifact("/tmp/MODEL_CARD.md", artifact_path="docs")

        # ── Register model ─────────────────────────────────────────────────────
        print("Logging MLflow PythonModel...")
        mlflow.pyfunc.log_model(
            artifact_path   = "model",
            python_model    = SREAssistantModel(),
            registered_model_name = MODEL_NAME,
            pip_requirements = ["ollama>=0.3.0", "pandas>=2.0.0"],
        )

        run_id = run.info.run_id
        print(f"\n✅ Model registered successfully")
        print(f"Run ID:      {run_id}")
        print(f"Experiment:  {EXPERIMENT_NAME}")
        print(f"Model:       {MODEL_NAME}")
        print(f"MLflow UI:   http://localhost:5000")

if __name__ == "__main__":
    register()