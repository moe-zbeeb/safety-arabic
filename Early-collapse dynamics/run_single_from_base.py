#!/usr/bin/env python3
"""
One-off: train from BASE directly to a single target step, eval it, clean up.
No base eval, no intermediate checkpoints.

Usage (for missing Llama-3 90/10 ckpt-30):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python /workspace/run_single_from_base.py \
            --base-model /workspace/models/Meta-Llama-3-8B-Instruct \
            --data-file /workspace/Data_train/90_safe_10_refusal.jsonl \
            --mix-name 90-10 \
            --target-step 30 \
            --output-dir /workspace/output_gpu0_single \
            --per-device-batch 4 --grad-accum 8

Note: --output-dir should be a FRESH dir (will be wiped if it exists). This is
to avoid any collisions with other ongoing workers' output_gpuN directories.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EVAL_SCRIPT  = Path("/workspace/Eval/eval.py")
TRAIN_SCRIPT = Path("/workspace/train_chunk.py")
LOG_DIR      = Path("/workspace/pipeline_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MAX_ATTEMPTS = 2


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
    p.add_argument("--base-model", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--mix-name", required=True)
    p.add_argument("--target-step", type=int, required=True,
                   help="Single target step to train to and eval (e.g. 30)")
    p.add_argument("--output-dir", required=True,
                   help="Fresh output dir — will be wiped if it exists")
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="Do not delete the checkpoint after eval")
    return p.parse_args()


def main():
    args = parse_args()

    base_model = Path(args.base_model)
    model_name = base_model.name
    run_name = f"{model_name}__{args.mix_name}"

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    worker_tag = f"gpu{gpu}__{model_name}__{args.mix_name}__fromBase{args.target_step}"
    log, log_file = make_logger(worker_tag)

    output_dir  = Path(args.output_dir)
    target_ckpt = output_dir / f"checkpoint-{args.target_step}"

    # Fresh output dir
    if output_dir.exists():
        log(f"Wiping existing output dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    log(f"========== SINGLE-FROM-BASE START ==========")
    log(f"Model:       {base_model}")
    log(f"Data:        {args.data_file}")
    log(f"Mix:         {args.mix_name}")
    log(f"Target:      checkpoint-{args.target_step}")
    log(f"Output dir:  {output_dir}")
    log(f"GPU:         {gpu}  batch={args.per_device_batch}  grad_accum={args.grad_accum}")

    # Sanity checks
    if not base_model.exists():
        log(f"ERROR: base model does not exist: {base_model}")
        sys.exit(1)
    if not Path(args.data_file).exists():
        log(f"ERROR: data file does not exist: {args.data_file}")
        sys.exit(1)

    # Train from base directly to target_step
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--base-model", str(base_model),
        "--data-file", args.data_file,
        "--output-dir", str(output_dir),
        "--target-step", str(args.target_step),
        "--per-device-batch-size", str(args.per_device_batch),
        "--grad-accum", str(args.grad_accum),
    ]
    if not run_with_retry(train_cmd, log_file, log, f"train base -> step {args.target_step}"):
        log(f"Training permanently failed. Exiting.")
        sys.exit(2)

    if not target_ckpt.exists():
        log(f"ERROR: expected checkpoint {target_ckpt} missing after training. Exiting.")
        sys.exit(3)

    # Eval target checkpoint
    eval_cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model", str(target_ckpt),
        "--run-name", run_name,
    ]
    eval_ok = run_with_retry(eval_cmd, log_file, log, f"eval ckpt-{args.target_step}")

    # Cleanup
    if eval_ok and not args.keep_checkpoint:
        log(f"Deleting checkpoint: {target_ckpt.name}")
        try:
            shutil.rmtree(target_ckpt)
        except Exception as e:
            log(f"WARN: failed to delete {target_ckpt}: {e}")
        # Also try to clean up the empty output dir
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            log(f"WARN: failed to clean up {output_dir}: {e}")
    elif not eval_ok:
        log(f"Eval failed — keeping checkpoint for manual retry.")
    else:
        log(f"--keep-checkpoint set — not deleting.")

    subprocess.run(["df", "-h", "/workspace"], check=False,
                   stdout=open(log_file, "a"), stderr=subprocess.STDOUT)
    log(f"========== SINGLE-FROM-BASE COMPLETE ==========")


if __name__ == "__main__":
    main()