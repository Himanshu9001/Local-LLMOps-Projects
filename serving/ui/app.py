"""
Phase 5 — Gradio Dashboard for SRE/DevOps AI Assistant
Run: python serving/ui/app.py
"""

import gradio as gr
import ollama

MODEL_NAME = "sre-assistant"

TEMPLATES = {
    "Pod CrashLoopBackOff":       "My Kubernetes pod is in CrashLoopBackOff. Exit code 137. Root cause and fix?",
    "Node NotReady":               "My AKS node is in NotReady state. Steps to diagnose?",
    "Terraform state lock":        "My Terraform apply is stuck on state lock. How do I safely unlock it?",
    "GitHub Actions OOM":          "My GitHub Actions runner is running out of memory. How do I debug?",
    "High pod memory":             "My pod memory usage keeps growing. How do I detect a memory leak in Kubernetes?",
    "Ingress 502 Bad Gateway":     "My Kubernetes ingress is returning 502 Bad Gateway. Root cause and fix?",
    "ArgoCD sync failed":          "My ArgoCD app is OutOfSync and sync is failing. Steps to debug?",
    "Prometheus high cardinality": "My Prometheus TSDB is running out of memory due to high cardinality. Fix?",
}

def query_model(question: str, context: str, history: list) -> tuple:
    if not question.strip():
        return history, "", context

    user_content = question.strip()
    if context.strip():
        user_content = f"Context/Logs:\n{context.strip()}\n\nQuestion: {user_content}"

    messages = []
    for human, assistant in history:
        messages.append({"role": "user",      "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_content})

    try:
        response = ollama.chat(
            model   = MODEL_NAME,
            messages = messages,
            options = {
                "temperature":    0.1,
                "top_p":          0.85,
                "repeat_penalty": 1.5,
                "num_predict":    400,
            }
        )
        answer = response["message"]["content"].strip()
    except Exception as e:
        answer = f"Error: {str(e)}\n\nMake sure Ollama is running: ollama serve"

    history.append((question, answer))
    return history, "", ""

def load_template(template_name: str) -> str:
    return TEMPLATES.get(template_name, "")

with gr.Blocks(title="SRE/DevOps AI Assistant") as demo:

    gr.Markdown("""
# ⚙ SRE/DevOps AI Assistant
**Mistral-7B · QLoRA Fine-tuned · Running locally via Ollama**  
Trained on: Kubernetes docs · AWS/Azure docs · Stack Overflow · GitHub postmortems
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="", height=450)

            question = gr.Textbox(
                placeholder = "e.g. My pod is OOMKilled with exit code 137. Root cause?",
                label       = "Question",
                lines       = 2,
            )

            context = gr.Textbox(
                placeholder = "Paste kubectl output, error logs, Terraform errors...",
                label       = "Context / Logs (optional)",
                lines       = 4,
            )

            with gr.Row():
                submit = gr.Button("▶  Ask", variant="primary", scale=3)
                clear  = gr.Button("✕  Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Quick Templates")
            gr.Markdown("<small>Click to load a template</small>")

            for name in TEMPLATES:
                btn = gr.Button(name, size="sm")
                btn.click(fn=lambda n=name: TEMPLATES[n], outputs=question)

            gr.Markdown("---")
            gr.Markdown("""
**Model:** Mistral-7B-Instruct-v0.3  
**Training:** QLoRA r=16, 700 steps  
**Dataset:** 7,302 examples  
**Quant:** Q4_K_M (4.17GB)  
**Serving:** Ollama · llama.cpp  
            """)

    submit.click(
        fn      = query_model,
        inputs  = [question, context, chatbot],
        outputs = [chatbot, question, context],
    )
    question.submit(
        fn      = query_model,
        inputs  = [question, context, chatbot],
        outputs = [chatbot, question, context],
    )
    clear.click(
        fn      = lambda: ([], "", ""),
        outputs = [chatbot, question, context],
    )

if __name__ == "__main__":
    print("Starting SRE/DevOps AI Assistant...")
    print("Open: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
