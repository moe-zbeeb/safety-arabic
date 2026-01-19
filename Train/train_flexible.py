#!/usr/bin/env python3
"""
Flexible training script for curriculum learning experiments.
Accepts command line arguments for model, dataset, and output directory.
"""

import argparse
import os
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the training dataset (jsonl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and final model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size (default: 16)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 60)
    print("CURRICULUM LEARNING TRAINING")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        "json",
        data_files=args.dataset_path,
        split="train"
    )
    print(f"Loaded {len(dataset)} examples")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Format dataset
    def format_chat(example):
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."},
                {"role": "user", "content": example["prompt_ar"]},
                {"role": "assistant", "content": example["response_ar"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            # For models without chat template, format manually
            text = f"أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا.\n\n{example['prompt_ar']}\n\n{example['response_ar']}"
        return {"text": text}

    print("Formatting dataset...")
    dataset = dataset.map(format_chat, num_proc=4)
    print(f"Formatted {len(dataset)} examples")

    # Training configuration
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=5,
        logging_steps=2,
        save_steps=args.save_steps,
        save_total_limit=None,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        fp16=True,
        max_grad_norm=1.0,
        seed=42,
        remove_unused_columns=True,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        logging_dir=logs_dir,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=args.model_path,
        args=config,
        train_dataset=dataset,
    )

    # Print training summary
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} examples")
    print(f"Model: {args.model_path}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save steps: {args.save_steps}")
    print("=" * 60 + "\n")

    # Train
    print("Starting training...\n")
    train_result = trainer.train()

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final-model")
    print(f"\nSaving final model to {final_model_path}...")
    trainer.save_model(final_model_path)

    # Print completion summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Model saved to: {final_model_path}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)

    # Save training info
    info_path = os.path.join(args.output_dir, "training_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Final loss: {train_result.training_loss:.4f}\n")
        f.write(f"Epochs: {args.num_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
    print(f"Training info saved to: {info_path}")


if __name__ == "__main__":
    main()
