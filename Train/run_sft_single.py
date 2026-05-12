#!/usr/bin/env python3
"""
Single-shot SFT trainer.

Runs one full training pass and saves ONE final checkpoint (no intermediate saves).
Based on the same setup that was used throughout EXP8, but without:
  - Multi-chunk scheduling
  - Periodic checkpointing
  - Per-checkpoint evaluation
  - Any orchestrator wrapping

Training config:
  - 1 epoch
  - Effective batch 32 (per_device_batch * grad_accum)
  - lr 1e-5, linear decay, 5 warmup steps
  - bf16, paged_adamw_8bit, gradient_checkpointing
  - attn_implementation="eager" (needed for Gemma2-family e.g. Fanar)
  - seed 42

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python run_sft_single.py \
            --base-model /workspace/models/Qwen2.5-3B-Instruct \
            --data-file /workspace/Data_train/100%refusals.jsonl \
            --output-dir /workspace/sft_out/qwen3b__100refusals \
            --per-device-batch 8 --grad-accum 4

Recommended batch settings (keep effective batch = 32):
    3B models:    --per-device-batch 8 --grad-accum 4
    7-8B models:  --per-device-batch 4 --grad-accum 8
    9B models:    --per-device-batch 2 --grad-accum 16

After training, the saved checkpoint will be at:
    {output-dir}/final/
with a .ready sentinel file confirming the save completed.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------- ARGS --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Single-shot SFT with final checkpoint only")
    p.add_argument("--base-model", required=True,
                   help="Path to base model (e.g. /workspace/models/Qwen2.5-3B-Instruct)")
    p.add_argument("--data-file", required=True,
                   help="Path to training JSONL with 'prompt_ar' and 'response_ar' fields")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the final checkpoint and logs")
    p.add_argument("--per-device-batch", type=int, default=4,
                   help="Per-device batch size (default 4)")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (default 8; effective batch = per-device * grad-accum)")
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Learning rate (default 1e-5)")
    p.add_argument("--warmup-steps", type=int, default=5,
                   help="Warmup steps (default 5)")
    p.add_argument("--max-length", type=int, default=512,
                   help="Max sequence length (default 512)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -------------------- MAIN --------------------

def main():
    args = parse_args()

    print(f"[sft] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[sft] base_model:       {args.base_model}")
    print(f"[sft] data_file:        {args.data_file}")
    print(f"[sft] output_dir:       {args.output_dir}")
    print(f"[sft] per_device_batch: {args.per_device_batch}")
    print(f"[sft] grad_accum:       {args.grad_accum}")
    print(f"[sft] effective_batch:  {args.per_device_batch * args.grad_accum}")
    print(f"[sft] learning_rate:    {args.learning_rate}")
    print(f"[sft] seed:             {args.seed}")

    # Validate inputs
    if not Path(args.base_model).exists():
        print(f"[sft] ERROR: base model path does not exist: {args.base_model}")
        sys.exit(1)
    if not Path(args.data_file).exists():
        print(f"[sft] ERROR: data file does not exist: {args.data_file}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print(f"[sft] Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Dataset
    print(f"[sft] Loading dataset from {args.data_file}")
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    print(f"[sft] Loaded {len(dataset)} examples")

    # Apply chat template if available, else fall back to a plain concat
    def format_chat(example):
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system",
                 "content": "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."},
                {"role": "user",      "content": example["prompt_ar"]},
                {"role": "assistant", "content": example["response_ar"]},
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=False)}
        return {"text": (
            "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا.\n\n"
            f"{example['prompt_ar']}\n\n{example['response_ar']}"
        )}

    dataset = dataset.map(format_chat, num_proc=4)

    # SFTConfig — 1 epoch, NO intermediate checkpoints (save_strategy="no"),
    # we save manually at the end via trainer.save_model(...)
    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        # Don't set max_steps; let 1 epoch drive total steps
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        logging_steps=2,
        save_strategy="no",            # NO intermediate saves
        save_total_limit=None,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=1.0,
        seed=args.seed,
        remove_unused_columns=True,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tb_logs"),
        max_length=args.max_length,
    )

    # Model
    print(f"[sft] Loading model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},            # relative to CUDA_VISIBLE_DEVICES
        attn_implementation="eager",   # needed for Gemma2-family (Fanar); harmless elsewhere
        trust_remote_code=True,
    )
    print(f"[sft] Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        # No processing_class/tokenizer arg — SFTTrainer will pull tokenizer from model
    )

    # Quick estimate of total steps for visibility
    n_examples = len(dataset)
    eff_batch  = args.per_device_batch * args.grad_accum
    total_steps = (n_examples + eff_batch - 1) // eff_batch  # ceil division
    print(f"[sft] Estimated total steps: {total_steps}  "
          f"({n_examples} examples / effective batch {eff_batch})")

    # Train
    print(f"[sft] Starting training...")
    result = trainer.train()
    print(f"[sft] Training done. Final loss: {result.training_loss:.4f}")

    # Save final checkpoint (model + tokenizer)
    final_dir = output_dir / "final"
    print(f"[sft] Saving final checkpoint to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Write ready sentinel (matches the convention from EXP8)
    ready_file = final_dir / ".ready"
    try:
        with open(ready_file, "w") as f:
            f.write(f"total_steps={result.global_step}\n")
            f.write(f"final_loss={result.training_loss:.6f}\n")
        print(f"[sft] Wrote ready marker: {ready_file}")
    except Exception as e:
        print(f"[sft] WARNING: failed to write ready marker: {e}")

    print(f"[sft] DONE")
    print(f"[sft]   Final checkpoint: {final_dir}")
    print(f"[sft]   To evaluate:")
    print(f"[sft]     python /workspace/Eval/eval.py --model {final_dir} "
          f"--run-name <your_run_name>")


if __name__ == "__main__":
    main()
