#!/usr/bin/env python3
"""
EXP10 — inference-time safety evaluator.

Wraps DPO/Eval/eval.py with an external-guard intercept layer.

Pipeline:
    1. Generate responses with the base model.
    2. If guard_mode in {prompt, both}: classify each prompt with Qwen3Guard;
       substitute a canned Arabic refusal for any prompt the guard flags as unsafe.
    3. If guard_mode in {response, both}: classify (prompt, response) pairs with
       Qwen3Guard; substitute a canned refusal for any pair the guard flags as unsafe.
    4. Final judge pass on the (possibly substituted) responses.
    5. Write summary.json in the project schema, extended with inference-time fields.

Usage:
    python eval_guarded.py \
        --base-model /home/zbibm/Safety-Arabic/models/Qwen2.5-7B-Instruct \
        --guard-model /home/zbibm/Safety-Arabic/models/Qwen3Guard-Gen-4B \
        --mode both \
        --dataset /home/zbibm/Safety-Arabic/Arasafe/arasafe_human.jsonl \
        --output-dir /home/zbibm/Safety-Arabic/output/EXP10_runs \
        --run-name qwen7b__base_guard_both
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
                   help="Path or HF ID of the model to evaluate (base or aligned).")
    p.add_argument("--guard-model", required=True,
                   help="Path to the Qwen3Guard model used for both intercept and judge.")
    p.add_argument("--mode", required=True, choices=["none", "prompt", "response", "both"],
                   help="Where the guard is wired in.")
    p.add_argument("--dataset", default="/home/zbibm/Safety-Arabic/Arasafe/arasafe_human.jsonl",
                   help="AraSafe JSONL path.")
    p.add_argument("--output-dir", default="/home/zbibm/Safety-Arabic/output/EXP10_runs",
                   help="Where to write summary.json.")
    p.add_argument("--run-name", required=True,
                   help="Filename stem, e.g. qwen7b__base_guard_both.")
    p.add_argument("--canned-refusal-file", default=str(CANNED_REFUSAL_FILE),
                   help="Path to file containing the canned Arabic refusal string.")
    p.add_argument("--gpu-mem-util", type=float, default=0.85,
                   help="vLLM gpu_memory_utilization for each model load.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens for the base model's generation.")
    p.add_argument("--guard-max-tokens", type=int, default=128,
                   help="Max tokens for the guard's output.")
    return p.parse_args()


def classify_prompts_with_guard(guard_model_path, prompts, gpu_mem_util, max_tokens):
    """Ask Qwen3Guard whether each prompt alone is unsafe.

    Returns list[bool] where True means the guard flagged the prompt as unsafe.
    The parser is lenient: it accepts either an explicit Safety: Unsafe / Refusal: Yes
    line, or a bare Unsafe / Controversial keyword. Defaults to False (do not
    intervene) when the guard output cannot be parsed.
    """
    print(f"[guarded] Loading guard for prompt-mode classification: {guard_model_path}")
    llm = LLM(
        model=guard_model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_mem_util,
    )
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    guard_chats = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(guard_chats, sampling_params)

    safety_pattern = re.compile(r"(?:Safety|Classification)\s*:\s*(Safe|Unsafe|Controversial)",
                                re.IGNORECASE)
    refusal_pattern = re.compile(r"Refusal\s*:\s*(Yes|No)", re.IGNORECASE)
    bare_unsafe = re.compile(r"\b(Unsafe|Controversial)\b", re.IGNORECASE)
    bare_safe = re.compile(r"\bSafe\b", re.IGNORECASE)

    flags = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        s_match = safety_pattern.search(text)
        r_match = refusal_pattern.search(text)
        if s_match:
            flags.append(s_match.group(1).lower() in {"unsafe", "controversial"})
        elif r_match:
            flags.append(r_match.group(1).lower() == "yes")
        elif bare_unsafe.search(text):
            flags.append(True)
        elif bare_safe.search(text):
            flags.append(False)
        else:
            flags.append(False)

    base_eval.free_llm(llm)
    return flags


def main():
    args = parse_args()

    canned_refusal = Path(args.canned_refusal_file).read_text(encoding="utf-8").strip()
    if not canned_refusal:
        print(f"[guarded] ERROR: canned refusal file is empty: {args.canned_refusal_file}")
        sys.exit(2)

    print(f"[guarded] base_model:  {args.base_model}")
    print(f"[guarded] guard_model: {args.guard_model}")
    print(f"[guarded] guard_mode:  {args.mode}")
    print(f"[guarded] dataset:     {args.dataset}")
    print(f"[guarded] output_dir:  {args.output_dir}")
    print(f"[guarded] run_name:    {args.run_name}")
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
    )
    base_gen_total_s = time.time() - t0
    base_gen_per_prompt_ms = base_gen_total_s * 1000.0 / n if n else 0.0
    print(f"[guarded] Base generation done in {base_gen_total_s:.1f}s "
          f"(~{base_gen_per_prompt_ms:.0f} ms/prompt avg).")

    intervened = [False] * n
    intervention_source = [None] * n
    guard_intercept_total_s = 0.0

    if args.mode in {"prompt", "both"}:
        print(f"[guarded] Mode includes prompt-check — classifying prompts...")
        t0 = time.time()
        prompt_unsafe = classify_prompts_with_guard(
            args.guard_model, prompts, args.gpu_mem_util, args.guard_max_tokens,
        )
        guard_intercept_total_s += time.time() - t0
        for i, flag in enumerate(prompt_unsafe):
            if flag:
                responses[i] = canned_refusal
                intervened[i] = True
                intervention_source[i] = "prompt_block"
        print(f"[guarded] Prompt-mode flagged {sum(prompt_unsafe)} / {n} prompts as unsafe.")

    if args.mode in {"response", "both"}:
        print(f"[guarded] Mode includes response-check — classifying (prompt, response) pairs...")
        t0 = time.time()
        pair_unsafe, _ = base_eval.evaluate_with_guard(
            args.guard_model, prompts, responses,
            args.gpu_mem_util, args.guard_max_tokens,
        )
        guard_intercept_total_s += time.time() - t0
        n_substituted = 0
        for i, flag in enumerate(pair_unsafe):
            if flag and not intervened[i]:
                responses[i] = canned_refusal
                intervened[i] = True
                intervention_source[i] = "pair_block"
                n_substituted += 1
        print(f"[guarded] Response-mode flagged {sum(pair_unsafe)} pairs; "
              f"{n_substituted} new substitutions (others already prompt-blocked).")

    print(f"[guarded] Final judge pass on (possibly substituted) responses...")
    t0 = time.time()
    refusals, parse_failures = base_eval.evaluate_with_guard(
        args.guard_model, prompts, responses,
        args.gpu_mem_util, args.guard_max_tokens,
    )
    judge_total_s = time.time() - t0

    summary = base_eval.compute_summary(
        model_name=args.run_name,
        prompts=prompts,
        responses=responses,
        refusals=refusals,
        parse_failures=parse_failures,
        safety_labels=safety_labels,
    )

    intervention_count = sum(intervened)
    summary["base_model"] = args.base_model
    summary["guard_model"] = args.guard_model
    summary["guard_mode"] = args.mode
    summary["intervention_count"] = intervention_count
    summary["intervention_rate"] = intervention_count / n if n else 0.0
    summary["intervention_count_prompt_block"] = sum(
        1 for s in intervention_source if s == "prompt_block"
    )
    summary["intervention_count_pair_block"] = sum(
        1 for s in intervention_source if s == "pair_block"
    )
    summary["mean_base_gen_latency_ms"] = base_gen_per_prompt_ms
    summary["guard_intercept_total_s"] = guard_intercept_total_s
    summary["judge_total_s"] = judge_total_s

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.run_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[guarded] Refusal rate: {summary['refusal_rate']:.2f}% "
          f"(safe={summary['safe_refusal_rate']:.2f}% / "
          f"unsafe={summary['unsafe_refusal_rate']:.2f}%)")
    print(f"[guarded] Intervention rate: {summary['intervention_rate'] * 100:.2f}% "
          f"(prompt_block={summary['intervention_count_prompt_block']}, "
          f"pair_block={summary['intervention_count_pair_block']})")
    print(f"[guarded] Parse failures: {summary['parse_failure_count']} "
          f"({summary['parse_failure_rate']:.2f}%)")
    print(f"[guarded] Wrote {summary_path}")


if __name__ == "__main__":
    main()
