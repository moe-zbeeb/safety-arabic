#!/usr/bin/env python3
"""
Single-GPU worker: process one (model, mix) combination end-to-end.
Base eval is SKIPPED in this version.

Flow:
  1. Train to step 5  (with base still on disk)
  2. Eval ckpt-5
  3. Delete base model from disk
  4. Train to step 10 (resume from ckpt-5)
  5. Eval ckpt-10
  6. Delete ckpt-5
  7. ...continue through step 100
  8. Delete final ckpt-100 after its eval

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_worker.py \
        --base-model /workspace/models/Qwen2.5-3B-Instruct \
        --data-file /workspace/Data_train/100%refusals.jsonl \
        --mix-name 100refusals \
        --per-device-batch 8 --grad-accum 4
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# -------------------- CONFIG --------------------

SAVE_STEPS   = [5, 10, 15, 20, 25, 30, 40, 60, 80, 100]
EVAL_SCRIPT  = Path("/workspace/Eval/eval.py")
TRAIN_SCRIPT = Path("/workspace/train_chunk.py")
LOG_DIR      = Path("/workspace/pipeline_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_ATTEMPTS = 2


# -------------------- HELPERS --------------------

def make_logger(worker_tag):
    log_file = LOG_DIR / f"{worker_tag}.log"

    def log(msg):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] [{worker_tag}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    return log, log_file


def run_subprocess(cmd, log_file, log, desc):
    """Run a command, streaming stdout/stderr to log_file. Return rc."""
    log(f"RUN: {desc}")
    with open(log_file, "a") as f:
        f.write(f"\n=== {desc} ===\n")
        f.write(f"CMD: {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc == 0:
        log(f"  OK: {desc}")
    else:
        log(f"  FAIL (rc={rc}): {desc}")
    return rc


def run_with_retry(cmd, log_file, log, desc):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rc = run_subprocess(cmd, log_file, log, f"{desc} (attempt {attempt})")
        if rc == 0:
            return True
    log(f"  GIVING UP on: {desc}")
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True,
                   help="Path to base model, e.g. /workspace/models/Qwen2.5-3B-Instruct")
    p.add_argument("--data-file", required=True,
                   help="Path to training JSONL, e.g. /workspace/Data_train/100%refusals.jsonl")
    p.add_argument("--mix-name", required=True,
                   help="Short tag for this data mix, e.g. '100refusals' or '95-5'")
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--output-root", default="/workspace",
                   help="Parent dir under which this worker creates its own output folder.")
    return p.parse_args()


# -------------------- MAIN --------------------

def main():
    args = parse_args()

    base_model = Path(args.base_model)
    model_name = base_model.name
    run_name = f"{model_name}__{args.mix_name}"

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    worker_tag = f"gpu{gpu}__{model_name}__{args.mix_name}"
    log, log_file = make_logger(worker_tag)

    # Each worker gets its own output dir to avoid collisions across parallel workers
    output_dir = Path(args.output_root) / f"output_gpu{gpu}"
    if output_dir.exists():
        log(f"Wiping existing output dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    log(f"========== WORKER START ==========")
    log(f"Model:    {base_model}")
    log(f"Data:     {args.data_file}")
    log(f"Mix:      {args.mix_name}")
    log(f"GPU:      {gpu}  batch={args.per_device_batch}  grad_accum={args.grad_accum}")
    log(f"Output:   {output_dir}")
    log(f"Run name: {run_name}")
    log(f"NOTE:     Base eval skipped in this version.")

    if not base_model.exists():
        log(f"ERROR: base model path does not exist. Exiting.")
        sys.exit(1)
    if not Path(args.data_file).exists():
        log(f"ERROR: data file does not exist. Exiting.")
        sys.exit(1)

    # ---------- Step 1: Train to first checkpoint (still have base on disk) ----------
    first_step = SAVE_STEPS[0]
    first_ckpt = output_dir / f"checkpoint-{first_step}"

    train_first_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--base-model", str(base_model),
        "--data-file", args.data_file,
        "--output-dir", str(output_dir),
        "--target-step", str(first_step),
        "--per-device-batch-size", str(args.per_device_batch),
        "--grad-accum", str(args.grad_accum),
    ]
    if not run_with_retry(train_first_cmd, log_file, log, f"train to step {first_step}"):
        log(f"First training chunk permanently failed. Exiting worker.")
        sys.exit(2)

    if not first_ckpt.exists():
        log(f"ERROR: expected checkpoint {first_ckpt} missing after training. Exiting.")
        sys.exit(3)

    # ---------- Step 2: Eval first checkpoint ----------
    eval_first_cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model", str(first_ckpt),
        "--run-name", run_name,
    ]
    run_with_retry(eval_first_cmd, log_file, log, f"eval ckpt-{first_step}")

    # ---------- Step 3: Delete base model ----------
    log(f"Deleting base model: {base_model}")
    try:
        shutil.rmtree(base_model)
    except Exception as e:
        log(f"WARN: failed to delete base model: {e}")
    subprocess.run(["df", "-h", "/workspace"], check=False,
                   stdout=open(log_file, "a"), stderr=subprocess.STDOUT)

    # ---------- Step 4: Loop through remaining checkpoints ----------
    prev_ckpt = first_ckpt

    for i in range(1, len(SAVE_STEPS)):
        step = SAVE_STEPS[i]
        ckpt_dir = output_dir / f"checkpoint-{step}"

        log(f"--- Checkpoint {step} ---")

        # Train resuming from prev_ckpt to this step
        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--base-model", str(base_model),  # unused when --resume-from is set, but required by arg
            "--data-file", args.data_file,
            "--output-dir", str(output_dir),
            "--target-step", str(step),
            "--resume-from", str(prev_ckpt),
            "--per-device-batch-size", str(args.per_device_batch),
            "--grad-accum", str(args.grad_accum),
        ]
        if not run_with_retry(train_cmd, log_file, log, f"train to step {step}"):
            log(f"Training to step {step} permanently failed. Exiting worker.")
            sys.exit(4)

        if not ckpt_dir.exists():
            log(f"ERROR: expected checkpoint {ckpt_dir} missing. Exiting.")
            sys.exit(5)

        # Eval this checkpoint
        eval_cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--model", str(ckpt_dir),
            "--run-name", run_name,
        ]
        run_with_retry(eval_cmd, log_file, log, f"eval ckpt-{step}")

        # Delete the previous checkpoint
        log(f"Deleting previous checkpoint: {prev_ckpt.name}")
        try:
            shutil.rmtree(prev_ckpt)
        except Exception as e:
            log(f"WARN: failed to delete {prev_ckpt}: {e}")

        prev_ckpt = ckpt_dir

    # ---------- Step 5: Delete the final checkpoint ----------
    log(f"Deleting final checkpoint: {prev_ckpt.name}")
    try:
        shutil.rmtree(prev_ckpt)
    except Exception as e:
        log(f"WARN: failed to delete {prev_ckpt}: {e}")

    # Clean up the whole output dir for this worker
    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        log(f"WARN: failed to clean up {output_dir}: {e}")

    subprocess.run(["df", "-h", "/workspace"], check=False,
                   stdout=open(log_file, "a"), stderr=subprocess.STDOUT)
    log(f"========== WORKER COMPLETE ==========")


if __name__ == "__main__":
    main()