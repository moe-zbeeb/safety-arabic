#!/usr/bin/env python3
"""
Single-shot DPO trainer.

Runs DPO on top of an SFT'd model and saves ONE final checkpoint.

Data format expected:
    JSON array. Each item has:
      - prompt_ar:        the (unsafe) prompt in Arabic
      - safe_response_ar: the preferred response (refusal) -> 'chosen'
      - response_ar:      the dispreferred response (compliant) -> 'rejected'

DPO config:
  - 1 epoch
  - Effective batch 32 (per_device_batch * grad_accum)
  - lr 5e-7 (DPO standard, 20-200x lower than SFT)
  - beta 0.1 (DPO temperature; default in TRL DPOTrainer)
  - bf16, paged_adamw_8bit, gradient_checkpointing
  - attn_implementation="eager" for cross-architecture safety
  - max_length=1024, max_prompt_length=512
  - seed 42

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python run_dpo_single.py \
            --base-model MHK-22/Qwen2.5-3B-SFT-safe \
            --data-file /workspace/Data_train/beavertails_safe_refusals_ar.json \
            --output-dir /workspace/dpo_out/qwen3b__beavertails \
            --per-device-batch 4 --grad-accum 8

Recommended batch settings (effective batch 32):
    3B models:   --per-device-batch 4 --grad-accum 8
    7-8B models: --per-device-batch 2 --grad-accum 16
    9B models:   --per-device-batch 1 --grad-accum 32

Note: DPO uses ~2x the GPU memory of SFT (loads both policy and reference model),
so per-device batch sizes are halved relative to the SFT script.

After training, the saved checkpoint will be at:
    {output-dir}/final/
with a .ready sentinel file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer


# -------------------- ARGS --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Single-shot DPO with final checkpoint only")
    p.add_argument("--base-model", required=True,
                   help="Path or HF repo of SFT'd model (e.g. MHK-22/Qwen2.5-3B-SFT-safe)")
    p.add_argument("--data-file", required=True,
                   help="Path to DPO JSON with prompt_ar, safe_response_ar, response_ar")
    p.add_argument("--output-dir", required=True,
                   help="Where to save the final checkpoint and logs")
    p.add_argument("--per-device-batch", type=int, default=2,
                   help="Per-device batch size (default 2 — DPO is memory-heavy)")
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation steps (default 16; effective batch 32)")
    p.add_argument("--learning-rate", type=float, default=5e-7,
                   help="Learning rate (default 5e-7 — standard for DPO)")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO beta (default 0.1 — TRL standard)")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--max-length", type=int, default=1024,
                   help="Max total sequence length")
    p.add_argument("--max-prompt-length", type=int, default=512,
                   help="Max prompt length")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -------------------- DATA LOADING --------------------

def load_dpo_dataset(path, tokenizer):
    """
    Load JSON file, convert to TRL DPO format with prompt/chosen/rejected.

    The prompt is formatted using the tokenizer's chat template (system + user)
    with add_generation_prompt=True so the model is positioned to generate.

    Chosen and rejected are the assistant's response strings only (no template).
    """
    print(f"[dpo] Loading data: {path}")
    with open(path) as f:
        raw = json.load(f)

    print(f"[dpo] Loaded {len(raw)} raw examples")

    # Filter examples with all required fields
    cleaned = []
    skipped = 0
    for item in raw:
        prompt_ar       = item.get("prompt_ar")
        safe_response   = item.get("safe_response_ar")
        unsafe_response = item.get("response_ar")
        if not (prompt_ar and safe_response and unsafe_response):
            skipped += 1
            continue
        cleaned.append({
            "prompt_ar": prompt_ar.strip(),
            "safe": safe_response.strip(),
            "unsafe": unsafe_response.strip(),
        })

    if skipped:
        print(f"[dpo] WARNING: skipped {skipped} examples missing required fields")
    print(f"[dpo] Usable examples: {len(cleaned)}")

    SYSTEM_PROMPT = "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."

    def build_record(ex):
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["prompt_ar"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,   # adds the assistant prefix so model generates
            )
        else:
            prompt_text = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{ex['prompt_ar']}\n\n"
            )
        return {
            "prompt":   prompt_text,
            "chosen":   ex["safe"],
            "rejected": ex["unsafe"],
        }

    records = [build_record(ex) for ex in cleaned]
    return Dataset.from_list(records)


# -------------------- MAIN --------------------

def main():
    args = parse_args()

    print(f"[dpo] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[dpo] base_model:        {args.base_model}")
    print(f"[dpo] data_file:         {args.data_file}")
    print(f"[dpo] output_dir:        {args.output_dir}")
    print(f"[dpo] per_device_batch:  {args.per_device_batch}")
    print(f"[dpo] grad_accum:        {args.grad_accum}")
    print(f"[dpo] effective_batch:   {args.per_device_batch * args.grad_accum}")
    print(f"[dpo] learning_rate:     {args.learning_rate}")
    print(f"[dpo] beta:              {args.beta}")
    print(f"[dpo] max_length:        {args.max_length}")
    print(f"[dpo] max_prompt_length: {args.max_prompt_length}")
    print(f"[dpo] seed:              {args.seed}")

    # Validate inputs
    if not Path(args.data_file).exists():
        print(f"[dpo] ERROR: data file does not exist: {args.data_file}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print(f"[dpo] Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Many SFT'd causal models don't have pad token; use EOS
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[dpo] tokenizer.pad_token was None — set to eos_token ({tokenizer.eos_token})")

    # Dataset
    train_dataset = load_dpo_dataset(args.data_file, tokenizer)
    print(f"[dpo] DPO dataset ready: {len(train_dataset)} pairs")
    print(f"[dpo] Sample record:")
    sample = train_dataset[0]
    print(f"        prompt[:200]:   {sample['prompt'][:200]!r}")
    print(f"        chosen[:200]:   {sample['chosen'][:200]!r}")
    print(f"        rejected[:200]: {sample['rejected'][:200]!r}")

    # Policy model (the one we're training)
    print(f"[dpo] Loading policy model from: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    print(f"[dpo] Policy model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # DPOConfig
    config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=2,
        save_strategy="no",  # NO intermediate saves
        save_total_limit=None,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=1.0,
        seed=args.seed,
        remove_unused_columns=False,  # IMPORTANT for DPO
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tb_logs"),
    )

    # Trainer — ref_model=None means TRL will create reference internally from policy
    # (using the same weights at start of training, kept frozen)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Step estimate
    n_examples = len(train_dataset)
    eff_batch  = args.per_device_batch * args.grad_accum
    total_steps = (n_examples + eff_batch - 1) // eff_batch
    print(f"[dpo] Estimated total steps: {total_steps}  "
          f"({n_examples} pairs / effective batch {eff_batch})")

    # Train
    print(f"[dpo] Starting DPO training...")
    result = trainer.train()
    print(f"[dpo] Training done. Final loss: {result.training_loss:.4f}")

    # Save final model + tokenizer
    final_dir = output_dir / "final"
    print(f"[dpo] Saving final checkpoint to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Ready sentinel
    ready_file = final_dir / ".ready"
    try:
        with open(ready_file, "w") as f:
            f.write(f"total_steps={result.global_step}\n")
            f.write(f"final_loss={result.training_loss:.6f}\n")
        print(f"[dpo] Wrote ready marker: {ready_file}")
    except Exception as e:
        print(f"[dpo] WARNING: failed to write ready marker: {e}")

    print(f"[dpo] DONE")
    print(f"[dpo]   Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
