"""
Fine-tuning Script — SRE/DevOps Mistral-7B QLoRA
Platform: Google Colab (T4 GPU) or any CUDA GPU
Usage:
    python training/finetune.py
    python training/finetune.py --resume          # resume from last checkpoint
    python training/finetune.py --steps 1230       # override total steps

Requirements:
    pip install "transformers==4.51.3" "datasets==3.4.1" "trl==0.18.2"
    pip install unsloth accelerate bitsandbytes peft
"""

import os
import argparse
import json
import torch
from pathlib import Path

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="QLoRA fine-tuning for SRE assistant")
parser.add_argument("--resume",       action="store_true", help="Resume from last checkpoint")
parser.add_argument("--dataset",      type=str, default="data/processed/train.jsonl")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--output_dir",   type=str, default="./final-model")
parser.add_argument("--hf_token",     type=str, default=os.environ.get("HF_TOKEN", ""))
parser.add_argument("--push_to_hub",  type=str, default="", help="HuggingFace repo to push to")
parser.add_argument("--epochs",       type=int, default=3)
parser.add_argument("--steps",        type=int, default=0, help="Max steps (0 = auto)")
args = parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL     = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT   = True

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0   # 0.0 enables Unsloth fast patching
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE  = 2e-4
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
WARMUP_STEPS   = 50
SAVE_STEPS     = 50
EVAL_STEPS     = 50
SAVE_LIMIT     = 3

# ── Environment check ──────────────────────────────────────────────────────────
def check_environment():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. This script requires a CUDA GPU.\n"
            "On Colab: Runtime → Change runtime type → T4 GPU"
        )
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU:  {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb < 14:
        print("WARNING: Less than 14GB VRAM — reduce MAX_SEQ_LENGTH if OOM occurs")

# ── Checkpoint detection ───────────────────────────────────────────────────────
def find_last_checkpoint(checkpoint_dir: str) -> str | None:
    path = Path(checkpoint_dir)
    if not path.exists():
        return None
    checkpoints = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )
    return str(checkpoints[-1]) if checkpoints else None

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SRE/DevOps AI Assistant — QLoRA Fine-tuning")
    print("=" * 60)

    # Environment check
    check_environment()

    # Detect checkpoint for resume
    last_checkpoint = None
    if args.resume:
        last_checkpoint = find_last_checkpoint(args.checkpoint_dir)
        if last_checkpoint:
            print(f"RESUME MODE — continuing from: {last_checkpoint}")
        else:
            print("No checkpoint found — starting fresh")
    else:
        print("FRESH START — training from step 0")

    # ── Load model ─────────────────────────────────────────────────────────────
    from unsloth import FastLanguageModel

    print("\nLoading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = LOAD_IN_4BIT,
    )
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── QLoRA adapters ─────────────────────────────────────────────────────────
    print("\nAttaching QLoRA adapters...")

    # If resuming, load existing adapters instead of initializing new ones
    if last_checkpoint and args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, last_checkpoint, is_trainable=True)
        print(f"Loaded adapters from: {last_checkpoint}")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r                          = LORA_R,
            target_modules             = LORA_TARGETS,
            lora_alpha                 = LORA_ALPHA,
            lora_dropout               = LORA_DROPOUT,
            bias                       = "none",
            use_gradient_checkpointing = "unsloth",
            random_state               = 42,
        )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total/1e6:.1f}M")
    print(f"Trainable params: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    from datasets import load_dataset

    if not Path(args.dataset).exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}\n"
            "Run data/prepare_dataset.py first to generate training data."
        )

    dataset       = load_dataset("json", data_files=args.dataset, split="train")
    dataset       = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset  = dataset["test"]

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ── Trainer ────────────────────────────────────────────────────────────────
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Clear Unsloth cache to prevent stale batch size
    cache_dir = Path("./unsloth_compiled_cache")
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("Cleared Unsloth compiled cache")

    max_steps = args.steps if args.steps > 0 else -1

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_dataset,
        eval_dataset       = eval_dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LENGTH,
        dataset_num_proc   = 2,
        args = TrainingArguments(
            per_device_train_batch_size  = BATCH_SIZE,
            gradient_accumulation_steps  = GRAD_ACCUM,
            warmup_steps                 = WARMUP_STEPS,
            num_train_epochs             = args.epochs,
            max_steps                    = max_steps,
            learning_rate                = LEARNING_RATE,
            fp16                         = True,
            bf16                         = False,
            logging_steps                = 10,
            eval_strategy                = "steps",
            eval_steps                   = EVAL_STEPS,
            save_strategy                = "steps",
            save_steps                   = SAVE_STEPS,
            save_total_limit             = SAVE_LIMIT,
            output_dir                   = args.checkpoint_dir,
            optim                        = "adamw_8bit",
            weight_decay                 = 0.01,
            lr_scheduler_type            = "cosine",
            seed                         = 42,
            report_to                    = "none",
            load_best_model_at_end       = True,
            metric_for_best_model        = "eval_loss",
            ddp_find_unused_parameters   = False,
        ),
    )

    # Safety check
    assert trainer.args.per_device_train_batch_size == BATCH_SIZE, \
        f"Batch size mismatch: expected {BATCH_SIZE}, got {trainer.args.per_device_train_batch_size}"
    print(f"Trainer ready — batch size: {BATCH_SIZE}, effective: {BATCH_SIZE * GRAD_ACCUM}")

    # ── Train ──────────────────────────────────────────────────────────────────
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * args.epochs
    print(f"\nStarting training...")
    print(f"Estimated steps:  {total_steps}")
    print(f"Checkpoint every: {SAVE_STEPS} steps → {args.checkpoint_dir}")
    print(f"Resuming from:    {last_checkpoint or 'scratch'}\n")

    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Steps:      {trainer_stats.global_step}")
    print(f"Train loss: {trainer_stats.training_loss:.4f}")
    print(f"{'='*60}")

    # ── Save merged model ──────────────────────────────────────────────────────
    print(f"\nMerging LoRA adapters → {args.output_dir}")
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"Merged model saved ✅")

    # ── Push to HuggingFace Hub ────────────────────────────────────────────────
    if args.push_to_hub and args.hf_token:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        from huggingface_hub import login
        login(token=args.hf_token)
        model.save_pretrained_merged(
            args.push_to_hub,
            tokenizer,
            save_method = "merged_16bit",
            push_to_hub = True,
            token       = args.hf_token,
        )
        print(f"Pushed to: https://huggingface.co/{args.push_to_hub} ✅")

    # ── Quick inference test ───────────────────────────────────────────────────
    print("\nRunning inference test...")
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        ["<s>[INST] You are an expert SRE engineer.\n\n"
         "My Kubernetes pod is in CrashLoopBackOff with Exit Code 137. "
         "What is the root cause and fix? [/INST]"],
        return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens  = 300,
        temperature     = 0.1,
        repetition_penalty = 1.3,
        do_sample       = True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest response:\n{response[200:]}\n")

    # Save training summary
    summary = {
        "base_model":        BASE_MODEL,
        "steps_completed":   trainer_stats.global_step,
        "train_loss":        trainer_stats.training_loss,
        "lora_r":            LORA_R,
        "lora_alpha":        LORA_ALPHA,
        "learning_rate":     LEARNING_RATE,
        "batch_size":        BATCH_SIZE,
        "grad_accum":        GRAD_ACCUM,
        "max_seq_length":    MAX_SEQ_LENGTH,
        "output_dir":        args.output_dir,
    }
    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved → training_summary.json")

if __name__ == "__main__":
    main()