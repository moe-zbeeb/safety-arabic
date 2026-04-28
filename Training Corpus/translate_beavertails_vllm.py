#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATASET = "PKU-Alignment/BeaverTails"
DEFAULT_MODEL = "hammh0a/Hala-1.2B-EN-AR-Translator"
DEFAULT_OUTPUT = "Training Corpus/beavertails_ar_10k_qa.jsonl"
TRANSLATION_INSTRUCTION = "Translate everything that follows into Arabic:\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate balanced BeaverTails QA pairs into Arabic with vLLM."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="330k_train")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--source", default="PKU-Alignment/BeaverTails translated with Hala")
    parser.add_argument("--no-shuffle-output", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def load_source_rows(dataset_name: str, split: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = []
    for index, row in enumerate(dataset):
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        categories = extract_categories(row.get("category"))
        if not prompt or not response or not categories:
            continue
        rows.append(
            {
                "source_index": index,
                "prompt_en": prompt,
                "response_en": response,
                "categories": categories,
                "is_safe": row.get("is_safe"),
            }
        )
    return rows


def extract_categories(value: object) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key, enabled in value.items() if enabled)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return sorted(str(item).strip() for item in value if str(item).strip())
    return []


def target_counts(categories: list[str], num_samples: int) -> dict[str, int]:
    if not categories:
        raise ValueError("No categories found")
    base = num_samples // len(categories)
    remainder = num_samples % len(categories)
    counts = {}
    for index, category in enumerate(sorted(categories)):
        counts[category] = base + (1 if index < remainder else 0)
    return counts


def balanced_sample(rows: list[dict], num_samples: int, seed: int) -> list[dict]:
    by_category = defaultdict(list)
    for row in rows:
        for category in row["categories"]:
            by_category[category].append(row)

    counts = target_counts(list(by_category), num_samples)
    selected = []
    used_source_indices = set()
    for category in sorted(by_category):
        category_rows = list(by_category[category])
        rng = random.Random(stable_seed(seed, category))
        rng.shuffle(category_rows)
        need = counts[category]
        unused_rows = [row for row in category_rows if row["source_index"] not in used_source_indices]
        picked = unused_rows[:need]
        if len(picked) < need:
            picked.extend(rng.choice(category_rows) for _ in range(need - len(picked)))
        for row in picked:
            copied = dict(row)
            copied["category"] = category
            selected.append(copied)
            used_source_indices.add(row["source_index"])
    return selected


def build_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": TRANSLATION_INSTRUCTION + text}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return TRANSLATION_INSTRUCTION + text + "\n\nArabic translation:"


def translate_texts(
    texts: list[str],
    model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int | None,
    max_new_tokens: int,
    trust_remote_code: bool,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt(tokenizer, text) for text in texts]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    source_rows = load_source_rows(args.dataset, args.split)
    if not source_rows:
        raise ValueError(f"No usable rows found in {args.dataset} split {args.split}")

    selected = balanced_sample(source_rows, args.num_samples, args.seed)
    if not args.no_shuffle_output:
        random.Random(args.seed).shuffle(selected)

    if args.plan_only:
        source_category_counts = Counter(category for row in source_rows for category in row["categories"])
        output_category_counts = Counter(row["category"] for row in selected)
        print(json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "num_requested_rows": args.num_samples,
                "num_source_rows": len(source_rows),
                "num_unique_selected_source_rows": len({row["source_index"] for row in selected}),
                "with_replacement": len(selected) > len({row["source_index"] for row in selected}),
                "source_category_counts": dict(sorted(source_category_counts.items())),
                "planned_output_category_counts": dict(sorted(output_category_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    all_translations = translate_texts(
        texts=[row["prompt_en"] for row in selected] + [row["response_en"] for row in selected],
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_translations = all_translations[:len(selected)]
    response_translations = all_translations[len(selected):]

    output_rows = []
    for sample_index, (row, prompt_ar, response_ar) in enumerate(zip(selected, prompt_translations, response_translations)):
        output_rows.append(
            {
                "prompt_ar": prompt_ar,
                "response_ar": response_ar,
                "prompt_en": row["prompt_en"],
                "response_en": row["response_en"],
                "category": row["category"],
                "categories": row["categories"],
                "is_safe": row["is_safe"],
                "source": args.source,
                "source_dataset": args.dataset,
                "source_split": args.split,
                "source_index": row["source_index"],
                "sample_index": sample_index,
                "translation_model": args.model,
            }
        )

    write_jsonl(args.output, output_rows)

    manifest_path = args.manifest if args.manifest else args.output.with_suffix(args.output.suffix + ".manifest.json")
    source_category_counts = Counter(category for row in source_rows for category in row["categories"])
    output_category_counts = Counter(row["category"] for row in output_rows)
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "translation_model": args.model,
        "output": str(args.output),
        "num_output_rows": len(output_rows),
        "num_source_rows": len(source_rows),
        "num_unique_selected_source_rows": len({row["source_index"] for row in selected}),
        "seed": args.seed,
        "source_category_counts": dict(sorted(source_category_counts.items())),
        "output_category_counts": dict(sorted(output_category_counts.items())),
        "with_replacement": len(output_rows) > len({row["source_index"] for row in selected}),
    }
    write_manifest(manifest_path, manifest)

    print(f"Wrote {len(output_rows)} rows to {args.output}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Translated {len(output_rows)} prompt-response pairs from {len(source_rows)} source rows")


if __name__ == "__main__":
    main()
