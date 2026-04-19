#!/usr/bin/env python3
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback, AutoModelForCausalLM

SAVE_STEPS = {5, 10, 15, 20, 25, 30, 40, 60, 80, 100}
MAX_STEPS = 100

MODEL_PATH = "/workspace/models/Fanar-1-9B"
DATA_FILE  = "/workspace/Data_train/100%refusals.jsonl"

print("Loading dataset...")
dataset = load_dataset(
    "json",
    data_files=DATA_FILE,
    split="train"
)
print(f"✓ Loaded {len(dataset)} examples")
print("Loading tokenizer and formatting dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
def format_chat(example):
    # chat templaet should be in tokenizer json files 
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
dataset = dataset.map(format_chat, num_proc=4)
print(f"✓ Formatted {len(dataset)} examples")

class CustomSaveStopCallback(TrainerCallback):
    """Save at specific steps and stop training at MAX_STEPS."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in SAVE_STEPS:
            control.should_save = True
        if state.global_step >= MAX_STEPS:
            control.should_save = True
            control.should_training_stop = True
        return control

config = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=2,          # was 16
    gradient_accumulation_steps=16,           # was 2  (effective batch still 32)
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=5,
    logging_steps=2,
    save_strategy="no",
    save_total_limit=None,
    completion_only_loss=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # added
    bf16=True,                               # was fp16=True
    max_grad_norm=1.0,
    seed=42,
    remove_unused_columns=True,
    dataloader_num_workers=2,
    report_to=["tensorboard"],
    logging_dir="./logs",
    max_length=512,
)

print("Loading model in bf16 on cuda:0...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="eager",
)
print(f">>> Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    callbacks=[CustomSaveStopCallback()],
)
print("\n" + "="*60)
print("="*60)
print(f"Dataset: {len(dataset)} examples")
print(f"Model: {MODEL_PATH}")
print(f"Data file: {DATA_FILE}")
print(f"Max steps: {MAX_STEPS} (incomplete cycle)")
print(f"Batch size: 2")
print(f"Gradient accumulation: 16")
print(f"Effective batch size: 32")
print(f"Learning rate: 1e-5")
print(f"Warmup steps: 5")
print(f"Save at steps: {sorted(SAVE_STEPS)}")
print("="*60 + "\n")
print("Starting training...\n")
train_result = trainer.train()
print("\nSaving model...")
trainer.save_model("./output/final-model")
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final loss: {train_result.training_loss:.4f}")
print(f"Model saved to: ./output/final-model")
print(f"Checkpoints saved at steps: {sorted(SAVE_STEPS)}")
print("="*60)