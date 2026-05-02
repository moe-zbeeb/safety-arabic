#!/usr/bin/env python3
"""
Sample 256 safe + 256 unsafe prompts for threshold calibration.

Sources
-------
  Unsafe: arasafe_synthetic_prompts.jsonl  (12,263 prompts, all unsafe)
  Safe:   hammh0a/Hala-4.6M-SFT (HuggingFace, streamed)

Length-matching
---------------
The arasafe synthetic prompts are tightly distributed (P5=36, P95=72, max=105).
We filter Hala safe prompts to the same [22, 105] character range so the
threshold isn't biased by length.

Output: data/threshold_safe.jsonl, data/threshold_unsafe.jsonl
  Schema: {"prompt": "...", "label": "normal" | "unsafe"}
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_PER_CLASS = 256

ARASAFE_PATH = Path("/Users/mohamad22/Downloads/arasafe_synthetic_prompts.jsonl")
HALA_DATASET = "hammh0a/Hala-4.6M-SFT"
HALA_OVERSAMPLE_TARGET = 4 * N_PER_CLASS  # collect a pool, then random-sample

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()) if s else ""


def percentile(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def sample_unsafe():
    with open(ARASAFE_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"AraSafe synthetic loaded: {len(records)}")

    prompts = [clean(r["Prompt"]) for r in records]
    prompts = [p for p in prompts if p]

    lens = sorted(len(p) for p in prompts)
    low = int(percentile(lens, 0))
    high = int(lens[-1])
    print(f"AraSafe length range: [{low}, {high}]  P5={percentile(lens, 5):.0f}  "
          f"P95={percentile(lens, 95):.0f}  mean={sum(lens)/len(lens):.0f}")

    rng = random.Random(SEED)
    sampled = rng.sample(prompts, N_PER_CLASS)
    return sampled, low, high


def sample_safe(low: int, high: int):
    rng = random.Random(SEED)
    print(f"\nStreaming {HALA_DATASET} (filter: {low} <= len <= {high} chars)...")

    ds = load_dataset(HALA_DATASET, split="train", streaming=True)

    pool = []
    seen = set()
    for ex in ds:
        conv = ex.get("conversations") or []
        if not conv:
            continue
        first = conv[0]
        if not isinstance(first, dict) or first.get("from") != "human":
            continue
        p = clean(first.get("value", ""))
        if not p or p in seen:
            continue
        if low <= len(p) <= high:
            pool.append(p)
            seen.add(p)
            if len(pool) >= HALA_OVERSAMPLE_TARGET:
                break

    print(f"Hala pool collected (length-filtered): {len(pool)}")
    if len(pool) < N_PER_CLASS:
        raise ValueError(
            f"Only collected {len(pool)} length-matched Hala prompts; need {N_PER_CLASS}. "
            f"Increase HALA_OVERSAMPLE_TARGET or loosen filter."
        )
    return rng.sample(pool, N_PER_CLASS)


def save(rows, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    unsafe_prompts, low, high = sample_unsafe()
    safe_prompts = sample_safe(low, high)

    safe_path = OUTPUT_DIR / "threshold_safe.jsonl"
    unsafe_path = OUTPUT_DIR / "threshold_unsafe.jsonl"

    save([{"prompt": p, "label": "normal"} for p in safe_prompts], safe_path)
    save([{"prompt": p, "label": "unsafe"} for p in unsafe_prompts], unsafe_path)

    safe_lens = [len(p) for p in safe_prompts]
    unsafe_lens = [len(p) for p in unsafe_prompts]
    print(f"\nWrote {len(safe_prompts)} safe   prompts to {safe_path}")
    print(f"Wrote {len(unsafe_prompts)} unsafe prompts to {unsafe_path}")
    print(f"Safe   chars  min={min(safe_lens)}  max={max(safe_lens)}  mean={sum(safe_lens)/len(safe_lens):.0f}")
    print(f"Unsafe chars  min={min(unsafe_lens)}  max={max(unsafe_lens)}  mean={sum(unsafe_lens)/len(unsafe_lens):.0f}")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()
