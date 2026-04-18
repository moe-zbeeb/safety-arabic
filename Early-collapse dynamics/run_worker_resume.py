#!/usr/bin/env python3
"""
Resume a run from a specific checkpoint.
Continues the train→eval→delete loop from the next SAVE_STEP onward.

Usage:
    CUDA_VISIBLE_DEVICES=2 python run_worker_resume.py \
        --base-model /workspace/models/Meta-Llama-3-8B-Instruct \
        --data-file /workspace/Data_train/95_safe_5_refusal.jsonl \
        --mix-name 95-5 \
        --resume-from-step 15 \
        --output-dir /workspace/output_gpu2 \
        --per-device-batch 4 --grad-accum 8
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SAVE_STEPS   = [5, 10, 15, 20, 25, 30, 40, 60, 80, 100]
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
    p.add_argument("--base-model", required=True,
                   help="Base model path (only used as arg passthrough to train_chunk.py)")
    p.add_argument("--data-file", required=True)
    p.add_argument("--mix-name", required=True)
    p.add_argument("--resume-from-step", type=int, required=True,
                   help="Step of the last successfully completed checkpoint (e.g. 15)")
    p.add_argument("--output-dir", required=True,
                   help="Existing output dir containing checkpoint-{resume-from-step}")
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume_from_step not in SAVE_STEPS:
        print(f"ERROR: --resume-from-step {args.resume_from_step} must be one of {SAVE_STEPS}")
        sys.exit(1)

    resume_idx = SAVE_STEPS.index(args.resume_from_step)
    remaining_steps = SAVE_STEPS[resume_idx + 1:]
    if not remaining_steps:
        print(f"Nothing to do — already at last step {args.resume_from_step}")
        sys.exit(0)

    base_model = Path(args.base_model)
    model_name = base_model.name
    run_name = f"{model_name}__{args.mix_name}"

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    worker_tag = f"gpu{gpu}__{model_name}__{args.mix_name}__resume{args.resume_from_step}"
    log, log_file = make_logger(worker_tag)

    output_dir = Path(args.output_dir)
    prev_ckpt = output_dir / f"checkpoint-{args.resume_from_step}"

    log(f"========== RESUME WORKER START ==========")
    log(f"Model:       {base_model}")
    log(f"Data:        {args.data_file}")
    log(f"Mix:         {args.mix_name}")
    log(f"Resume from: checkpoint-{args.resume_from_step}")
    log(f"Remaining:   {remaining_steps}")
    log(f"Output dir:  {output_dir}  (NOT wiping!)")

    if not prev_ckpt.exists():
        log(f"ERROR: resume checkpoint {prev_ckpt} does not exist. Exiting.")
        sys.exit(1)
    if not (prev_ckpt / ".ready").exists():
        log(f"ERROR: resume checkpoint missing .ready marker. It may be corrupt. Exiting.")
        sys.exit(1)

    # Loop through remaining checkpoints
    for step in remaining_steps:
        ckpt_dir = output_dir / f"checkpoint-{step}"
        log(f"--- Checkpoint {step} ---")

        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--base-model", str(base_model),  # unused when --resume-from is set
            "--data-file", args.data_file,
            "--output-dir", str(output_dir),
            "--target-step", str(step),
            "--resume-from", str(prev_ckpt),
            "--per-device-batch-size", str(args.per_device_batch),
            "--grad-accum", str(args.grad_accum),
        ]
        if not run_with_retry(train_cmd, log_file, log, f"train to step {step}"):
            log(f"Training to step {step} permanently failed. Exiting.")
            sys.exit(2)

        if not ckpt_dir.exists():
            log(f"ERROR: expected checkpoint {ckpt_dir} missing. Exiting.")
            sys.exit(3)

        eval_cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--model", str(ckpt_dir),
            "--run-name", run_name,
        ]
        run_with_retry(eval_cmd, log_file, log, f"eval ckpt-{step}")

        log(f"Deleting previous checkpoint: {prev_ckpt.name}")
        try:
            shutil.rmtree(prev_ckpt)
        except Exception as e:
            log(f"WARN: failed to delete {prev_ckpt}: {e}")

        prev_ckpt = ckpt_dir

    log(f"Deleting final checkpoint: {prev_ckpt.name}")
    try:
        shutil.rmtree(prev_ckpt)
    except Exception as e:
        log(f"WARN: failed to delete {prev_ckpt}: {e}")

    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        log(f"WARN: failed to clean up {output_dir}: {e}")

    subprocess.run(["df", "-h", "/workspace"], check=False,
                   stdout=open(log_file, "a"), stderr=subprocess.STDOUT)
    log(f"========== RESUME WORKER COMPLETE ==========")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Resume a run from a specific checkpoint.
Continues the train→eval→delete loop from the next SAVE_STEP onward.

Usage:
    CUDA_VISIBLE_DEVICES=2 python run_worker_resume.py \
        --base-model /workspace/models/Meta-Llama-3-8B-Instruct \
        --data-file /workspace/Data_train/95_safe_5_refusal.jsonl \
        --mix-name 95-5 \
        --resume-from-step 15 \
        --output-dir /workspace/output_gpu2 \
        --per-device-batch 4 --grad-accum 8
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SAVE_STEPS   = [5, 10, 15, 20, 25, 30, 40, 60, 80, 100]
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
    p.add_argument("--base-model", required=True,
                   help="Base model path (only used as arg passthrough to train_chunk.py)")
    p.add_argument("--data-file", required=True)
    p.add_argument("--mix-name", required=True)
    p.add_argument("--resume-from-step", type=int, required=True,
                   help="Step of the last successfully completed checkpoint (e.g. 15)")
    p.add_argument("--output-dir", required=True,
                   help="Existing output dir containing checkpoint-{resume-from-step}")
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume_from_step not in SAVE_STEPS:
        print(f"ERROR: --resume-from-step {args.resume_from_step} must be one of {SAVE_STEPS}")
        sys.exit(1)

    resume_idx = SAVE_STEPS.index(args.resume_from_step)
    remaining_steps = SAVE_STEPS[resume_idx + 1:]
    if not remaining_steps:
        print(f"Nothing to do — already at last step {args.resume_from_step}")
        sys.exit(0)

    base_model = Path(args.base_model)
    model_name = base_model.name
    run_name = f"{model_name}__{args.mix_name}"

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    worker_tag = f"gpu{gpu}__{model_name}__{args.mix_name}__resume{args.resume_from_step}"
    log, log_file = make_logger(worker_tag)

    output_dir = Path(args.output_dir)
    prev_ckpt = output_dir / f"checkpoint-{args.resume_from_step}"

    log(f"========== RESUME WORKER START ==========")
    log(f"Model:       {base_model}")
    log(f"Data:        {args.data_file}")
    log(f"Mix:         {args.mix_name}")
    log(f"Resume from: checkpoint-{args.resume_from_step}")
    log(f"Remaining:   {remaining_steps}")
    log(f"Output dir:  {output_dir}  (NOT wiping!)")

    if not prev_ckpt.exists():
        log(f"ERROR: resume checkpoint {prev_ckpt} does not exist. Exiting.")
        sys.exit(1)
    if not (prev_ckpt / ".ready").exists():
        log(f"ERROR: resume checkpoint missing .ready marker. It may be corrupt. Exiting.")
        sys.exit(1)

    # Loop through remaining checkpoints
    for step in remaining_steps:
        ckpt_dir = output_dir / f"checkpoint-{step}"
        log(f"--- Checkpoint {step} ---")

        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--base-model", str(base_model),  # unused when --resume-from is set
            "--data-file", args.data_file,
            "--output-dir", str(output_dir),
            "--target-step", str(step),
            "--resume-from", str(prev_ckpt),
            "--per-device-batch-size", str(args.per_device_batch),
            "--grad-accum", str(args.grad_accum),
        ]
        if not run_with_retry(train_cmd, log_file, log, f"train to step {step}"):
            log(f"Training to step {step} permanently failed. Exiting.")
            sys.exit(2)

        if not ckpt_dir.exists():
            log(f"ERROR: expected checkpoint {ckpt_dir} missing. Exiting.")
            sys.exit(3)

        eval_cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--model", str(ckpt_dir),
            "--run-name", run_name,
        ]
        run_with_retry(eval_cmd, log_file, log, f"eval ckpt-{step}")

        log(f"Deleting previous checkpoint: {prev_ckpt.name}")
        try:
            shutil.rmtree(prev_ckpt)
        except Exception as e:
            log(f"WARN: failed to delete {prev_ckpt}: {e}")

        prev_ckpt = ckpt_dir

    log(f"Deleting final checkpoint: {prev_ckpt.name}")
    try:
        shutil.rmtree(prev_ckpt)
    except Exception as e:
        log(f"WARN: failed to delete {prev_ckpt}: {e}")

    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        log(f"WARN: failed to clean up {output_dir}: {e}")

    subprocess.run(["df", "-h", "/workspace"], check=False,
                   stdout=open(log_file, "a"), stderr=subprocess.STDOUT)
    log(f"========== RESUME WORKER COMPLETE ==========")


if __name__ == "__main__":
    main()