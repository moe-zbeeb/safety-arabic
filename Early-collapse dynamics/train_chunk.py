#!/usr/bin/env python3
"""
Train a single chunk of steps with resume-from-checkpoint support.
Writes a `.ready` sentinel file inside the saved checkpoint after save completes.
"""
import argparse
import os
import sys

# CUDA_VISIBLE_DEVICES must be set BEFORE importing torch.
# The orchestrator sets it via env; this script just respects it.

import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback


MAX_STEPS = 100  # LR schedule horizon; never change mid-run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-step", type=int, required=True)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[train] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    print(f"[train] base_model={args.base_model}")
    print(f"[train] data_file={args.data_file}")
    print(f"[train] target_step={args.target_step}  resume_from={args.resume_from}")
    print(f"[train] per_device_batch={args.per_device_batch_size}  grad_accum={args.grad_accum}")

    tokenizer_source = args.resume_from if args.resume_from else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    dataset = load_dataset("json", data_files=args.data_file, split="train")
    print(f"[train] Loaded {len(dataset)} examples")

    def format_chat(example):
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."},
                {"role": "user", "content": example["prompt_ar"]},
                {"role": "assistant", "content": example["response_ar"]},
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
        return {"text": f"أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا.\n\n{example['prompt_ar']}\n\n{example['response_ar']}"}

    dataset = dataset.map(format_chat, num_proc=4)

    target = args.target_step
    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{target}")

    class ChunkCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            if state.global_step >= target:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args_, state, control, **kwargs):
            # Called AFTER the checkpoint is written. Drop a ready marker.
            ready_file = os.path.join(ckpt_path, ".ready")
            try:
                with open(ready_file, "w") as f:
                    f.write(f"step={state.global_step}\n")
                print(f"[train] Wrote ready marker: {ready_file}")
            except Exception as e:
                print(f"[train] WARNING: failed to write ready marker: {e}")
            return control

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_steps=5,
        logging_steps=2,
        save_strategy="no",
        save_total_limit=None,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=1.0,
        seed=42,
        remove_unused_columns=True,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "logs"),
        max_length=512,
    )

    model_source = args.resume_from if args.resume_from else args.base_model
    print(f"[train] Loading model from: {model_source}")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},          # relative to CUDA_VISIBLE_DEVICES
        attn_implementation="eager",
    )
    print(f"[train] Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        callbacks=[ChunkCallback()],
    )

    print(f"[train] Training to step {target}...")
    if args.resume_from:
        result = trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        result = trainer.train()
    print(f"[train] Done. loss={result.training_loss:.4f}")


if __name__ == "__main__":
    main()


