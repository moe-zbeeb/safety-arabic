#!/usr/bin/env python3
"""
Inference-time safety evaluator with separate guard and judge.

Architecture:
    - GUARD (default: Llama-Guard-3-8B) intercepts the model's response in real
      time. If the (prompt, response) pair is flagged unsafe, the response is
      replaced with a canned Arabic refusal before it ever reaches the user.
    - JUDGE (default: Qwen3Guard-Gen-4B) scores the user-facing response after
      the guard has done its work. This is the same scorer used in EXP2/EXP3/
      EXP6/EXP9, so the rates are directly comparable.

Pipeline:
    1. Generate responses with the base/SFT model.
    2. response/both modes: feed (prompt, response) pairs to the GUARD; for any
       pair flagged unsafe, substitute the canned refusal.
       prompt/both modes: feed prompts alone to the GUARD; for any prompt
       flagged unsafe, substitute the canned refusal up front.
    3. Score every (prompt, final response) pair with the JUDGE.
    4. Compute the standard summary fields plus the new inference-time fields.

Usage:
    python eval_guarded.py \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --guard-model meta-llama/Llama-Guard-3-8B \
        --judge-model Qwen/Qwen3Guard-Gen-4B \
        --mode response \
        --dataset /workspace/Safety-Arabic/Arasafe/arasafe_human.jsonl \
        --output-dir /workspace/Safety-Arabic/output/exp7_base_sft_guard \
        --run-name llama_base__base_guard_response
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "DPO" / "Eval"))
import eval as base_eval  # noqa: E402


CANNED_REFUSAL_FILE = Path(__file__).resolve().parent / "canned_refusal_ar.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Inference-time guarded AraSafe eval.")
    p.add_argument("--base-model", required=True,
                   help="Path or HF ID of the model under test (base or SFT/aligned).")
    p.add_argument("--tokenizer-model", default=None,
                   help="Optional tokenizer path or HF ID override for the model under test. "
                        "Useful when an SFT repo has broken tokenizer metadata but still uses "
                        "the base model tokenizer.")
    p.add_argument("--guard-model", required=True,
                   help="Path or HF ID of the GUARD that intercepts at inference time. "
                        "Default expectation is a Llama-Guard model.")
    p.add_argument("--judge-model", default="Qwen/Qwen3Guard-Gen-4B",
                   help="Path or HF ID of the JUDGE that scores the final response. "
                        "Default is Qwen3Guard-Gen-4B (the project's standard judge).")
    p.add_argument("--guard-family", default="auto",
                   choices=["auto", "llamaguard", "qwen3guard"],
                   help="Output parser family for the GUARD. 'auto' picks based on "
                        "the model path/name.")
    p.add_argument("--mode", required=True, choices=["none", "prompt", "response", "both"],
                   help="Where the guard is wired in.")
    p.add_argument("--dataset", default="/home/zbibm/Safety-Arabic/Arasafe/arasafe_human.jsonl",
                   help="AraSafe JSONL path.")
    p.add_argument("--output-dir", default="/home/zbibm/Safety-Arabic/output/EXP10_runs",
                   help="Where to write summary.json.")
    p.add_argument("--run-name", required=True,
                   help="Filename stem, e.g. llama_base__base_guard_response.")
    p.add_argument("--canned-refusal-file", default=str(CANNED_REFUSAL_FILE),
                   help="Path to file containing the canned Arabic refusal string.")
    p.add_argument("--gpu-mem-util", type=float, default=0.85,
                   help="vLLM gpu_memory_utilization for each model load.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens for the base model's generation.")
    p.add_argument("--guard-max-tokens", type=int, default=128,
                   help="Max tokens for the guard's output.")
    return p.parse_args()


def detect_guard_family(family_arg, guard_model_path):
    if family_arg != "auto":
        return family_arg
    name = guard_model_path.lower()
    if "llama-guard" in name or "llamaguard" in name or "llama_guard" in name:
        return "llamaguard"
    return "qwen3guard"


def parse_llamaguard_verdict(text):
    first = next((l.strip() for l in text.strip().splitlines() if l.strip()), "")
    low = first.lower()
    if low.startswith("unsafe"):
        return True, False
    if low.startswith("safe"):
        return False, False
    return False, True


def parse_qwen3guard_verdict(text):
    safety_pattern = re.compile(
        r"(?:Safety|Classification)\s*:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE
    )
    refusal_pattern = re.compile(r"Refusal\s*:\s*(Yes|No)", re.IGNORECASE)
    s_match = safety_pattern.search(text)
    if s_match:
        return s_match.group(1).lower() in {"unsafe", "controversial"}, False
    r_match = refusal_pattern.search(text)
    if r_match:
        return r_match.group(1).lower() == "yes", False
    if re.search(r"\b(unsafe|controversial)\b", text, re.IGNORECASE):
        return True, False
    if re.search(r"\bsafe\b", text, re.IGNORECASE):
        return False, False
    return False, True


def classify_with_guard(guard_model_path, chats, gpu_mem_util, max_tokens, family):
    print(f"[guarded] Loading {family} guard: {guard_model_path}")
    llm = LLM(
        model=guard_model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_mem_util,
    )
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    outputs = llm.chat(chats, sampling_params)

    flags = []
    parse_failures = []
    parser = parse_llamaguard_verdict if family == "llamaguard" else parse_qwen3guard_verdict
    for output in outputs:
        text = output.outputs[0].text.strip()
        is_unsafe, parse_failed = parser(text)
        flags.append(is_unsafe)
        parse_failures.append(parse_failed)

    base_eval.free_llm(llm)
    return flags, parse_failures


def main():
    args = parse_args()

    canned_refusal = Path(args.canned_refusal_file).read_text(encoding="utf-8").strip()
    if not canned_refusal:
        print(f"[guarded] ERROR: canned refusal file is empty: {args.canned_refusal_file}")
        sys.exit(2)

    guard_family = detect_guard_family(args.guard_family, args.guard_model)

    print(f"[guarded] base_model:    {args.base_model}")
    print(f"[guarded] tokenizer:     {args.tokenizer_model or args.base_model}")
    print(f"[guarded] guard_model:   {args.guard_model}  (family={guard_family})")
    print(f"[guarded] judge_model:   {args.judge_model}")
    print(f"[guarded] guard_mode:    {args.mode}")
    print(f"[guarded] dataset:       {args.dataset}")
    print(f"[guarded] output_dir:    {args.output_dir}")
    print(f"[guarded] run_name:      {args.run_name}")
    print(f"[guarded] canned refusal ({len(canned_refusal)} chars): {canned_refusal!r}")

    print(f"[guarded] Loading dataset...")
    dataset = base_eval.load_dataset(args.dataset)
    prompts = [item["User_Prompt"] for item in dataset]
    safety_labels = [item["Safety_Label"] for item in dataset]
    n = len(prompts)
    print(f"[guarded] {n} prompts loaded.")

    print(f"[guarded] Generating base model responses...")
    t0 = time.time()
    responses = base_eval.generate_responses(
        model_path=args.base_model,
        prompts=prompts,
        gpu_mem_util=args.gpu_mem_util,
        max_new_tokens=args.max_new_tokens,
        tokenizer_path=args.tokenizer_model,
    )
    base_gen_total_s = time.time() - t0
    base_gen_per_prompt_ms = base_gen_total_s * 1000.0 / n if n else 0.0
    print(f"[guarded] Base generation done in {base_gen_total_s:.1f}s "
          f"(~{base_gen_per_prompt_ms:.0f} ms/prompt avg).")

    intervened = [False] * n
    intervention_source = [None] * n
    guard_intercept_total_s = 0.0
    guard_parse_failure_count = 0

    if args.mode in {"prompt", "both"}:
        print(f"[guarded] Mode includes prompt-check — guarding prompts...")
        prompt_chats = [[{"role": "user", "content": p}] for p in prompts]
        t0 = time.time()
        prompt_unsafe, prompt_parse_failures = classify_with_guard(
            args.guard_model, prompt_chats, args.gpu_mem_util,
            args.guard_max_tokens, guard_family,
        )
        guard_intercept_total_s += time.time() - t0
        guard_parse_failure_count += sum(prompt_parse_failures)
        for i, flag in enumerate(prompt_unsafe):
            if flag:
                responses[i] = canned_refusal
                intervened[i] = True
                intervention_source[i] = "prompt_block"
        print(f"[guarded] Prompt guard flagged {sum(prompt_unsafe)} / {n} prompts as unsafe.")

    if args.mode in {"response", "both"}:
        print(f"[guarded] Mode includes response-check — guarding (prompt, response) pairs...")
        pair_chats = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(prompts, responses)
        ]
        t0 = time.time()
        pair_unsafe, pair_parse_failures = classify_with_guard(
            args.guard_model, pair_chats, args.gpu_mem_util,
            args.guard_max_tokens, guard_family,
        )
        guard_intercept_total_s += time.time() - t0
        guard_parse_failure_count += sum(pair_parse_failures)
        n_substituted = 0
        for i, flag in enumerate(pair_unsafe):
            if flag and not intervened[i]:
                responses[i] = canned_refusal
                intervened[i] = True
                intervention_source[i] = "pair_block"
                n_substituted += 1
        print(f"[guarded] Response guard flagged {sum(pair_unsafe)} pairs; "
              f"{n_substituted} new substitutions (others already prompt-blocked).")

    print(f"[guarded] Final JUDGE pass on (possibly substituted) responses...")
    t0 = time.time()
    refusals, judge_parse_failures = base_eval.evaluate_with_guard(
        args.judge_model, prompts, responses,
        args.gpu_mem_util, args.guard_max_tokens,
    )
    judge_total_s = time.time() - t0

    summary = base_eval.compute_summary(
        model_name=args.run_name,
        prompts=prompts,
        responses=responses,
        refusals=refusals,
        parse_failures=judge_parse_failures,
        safety_labels=safety_labels,
    )

    intervention_count = sum(intervened)
    summary["base_model"] = args.base_model
    summary["tokenizer_model"] = args.tokenizer_model or args.base_model
    summary["guard_model"] = args.guard_model
    summary["guard_family"] = guard_family
    summary["judge_model"] = args.judge_model
    summary["guard_mode"] = args.mode
    summary["intervention_count"] = intervention_count
    summary["intervention_rate"] = intervention_count / n if n else 0.0
    summary["intervention_count_prompt_block"] = sum(
        1 for s in intervention_source if s == "prompt_block"
    )
    summary["intervention_count_pair_block"] = sum(
        1 for s in intervention_source if s == "pair_block"
    )
    summary["guard_parse_failure_count"] = guard_parse_failure_count
    summary["guard_parse_failure_rate"] = (
        guard_parse_failure_count / n if n else 0.0
    )
    summary["mean_base_gen_latency_ms"] = base_gen_per_prompt_ms
    summary["guard_intercept_total_s"] = guard_intercept_total_s
    summary["judge_total_s"] = judge_total_s

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.run_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[guarded] Final refusal rate: {summary['refusal_rate']:.2f}% "
          f"(safe={summary['safe_refusal_rate']:.2f}% / "
          f"unsafe={summary['unsafe_refusal_rate']:.2f}%)")
    print(f"[guarded] Guard intervention rate: {summary['intervention_rate'] * 100:.2f}% "
          f"(prompt_block={summary['intervention_count_prompt_block']}, "
          f"pair_block={summary['intervention_count_pair_block']})")
    print(f"[guarded] Guard parse failures: {guard_parse_failure_count} "
          f"({summary['guard_parse_failure_rate'] * 100:.2f}%)")
    print(f"[guarded] Judge parse failures: {summary['parse_failure_count']} "
          f"({summary['parse_failure_rate']:.2f}%)")
    print(f"[guarded] Wrote {summary_path}")


if __name__ == "__main__":
    main()
