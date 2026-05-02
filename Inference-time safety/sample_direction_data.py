#!/usr/bin/env python3
"""
Sample 1000 safe + 1000 unsafe prompts for recomputing the safety direction.

Source: dpo_20k_safe_first.jsonl
  - First 10,000 records: safe prompts
  - Last 10,000 records: unsafe prompts

The two pools have wildly different length distributions (safe ~575 chars mean,
unsafe ~62 chars mean). To prevent the safety direction from encoding "long vs
short" instead of "safe vs unsafe", we:
  1. Measure the unsafe length distribution.
  2. Filter the safe pool to the same character-length range as unsafe.
  3. Sample 1000 from each filtered pool.

Output: data/direction_safe.jsonl, data/direction_unsafe.jsonl
  Schema: {"prompt": "...", "label": "normal" | "unsafe"}
"""

import json
import random
from pathlib import Path

SEED = 42
N_PER_CLASS = 1000
SAFE_SPLIT_END = 10_000  # first 10k are safe
# Bound the safe pool to the central mass of the unsafe distribution
LEN_LOW_PCTL = 5
LEN_HIGH_PCTL = 95

SOURCE_PATH = Path(
    "/Users/mohamad22/Desktop/DPO/Data_for_DPO/final_res/dpo_20k_safe_first.jsonl"
)
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(SEED)

    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 20_000, f"Expected 20k records, got {len(records)}"

    safe_pool = records[:SAFE_SPLIT_END]
    unsafe_pool = records[SAFE_SPLIT_END:]

    unsafe_lens_all = sorted(len(r["prompt"]) for r in unsafe_pool)

    def percentile(sorted_vals, p):
        k = (len(sorted_vals) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

    low = int(percentile(unsafe_lens_all, LEN_LOW_PCTL))
    high = int(percentile(unsafe_lens_all, LEN_HIGH_PCTL))
    print(f"Unsafe pool length: full range [{unsafe_lens_all[0]}, {unsafe_lens_all[-1]}], "
          f"P{LEN_LOW_PCTL}-P{LEN_HIGH_PCTL} = [{low}, {high}] (n={len(unsafe_pool)})")

    unsafe_pool_filtered = [r for r in unsafe_pool if low <= len(r["prompt"]) <= high]
    safe_pool_filtered = [r for r in safe_pool if low <= len(r["prompt"]) <= high]
    print(f"Unsafe pool after length filter [{low}, {high}]: "
          f"{len(unsafe_pool_filtered)} / {len(unsafe_pool)} kept")
    print(f"Safe pool after length filter   [{low}, {high}]: "
          f"{len(safe_pool_filtered)} / {len(safe_pool)} kept")

    for name, pool in [("safe", safe_pool_filtered), ("unsafe", unsafe_pool_filtered)]:
        if len(pool) < N_PER_CLASS:
            raise ValueError(
                f"Filtered {name} pool only has {len(pool)} prompts, need {N_PER_CLASS}. "
                f"Loosen LEN_LOW_PCTL / LEN_HIGH_PCTL."
            )

    safe_idx = random.sample(range(len(safe_pool_filtered)), N_PER_CLASS)
    unsafe_idx = random.sample(range(len(unsafe_pool_filtered)), N_PER_CLASS)

    safe_prompts = [{"prompt": safe_pool_filtered[i]["prompt"], "label": "normal"} for i in safe_idx]
    unsafe_prompts = [{"prompt": unsafe_pool_filtered[i]["prompt"], "label": "unsafe"} for i in unsafe_idx]

    safe_path = OUTPUT_DIR / "direction_safe.jsonl"
    unsafe_path = OUTPUT_DIR / "direction_unsafe.jsonl"

    for path, rows in [(safe_path, safe_prompts), (unsafe_path, unsafe_prompts)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    safe_lens = [len(r["prompt"]) for r in safe_prompts]
    unsafe_lens = [len(r["prompt"]) for r in unsafe_prompts]

    print(f"Wrote {len(safe_prompts)} safe prompts to {safe_path}")
    print(f"Wrote {len(unsafe_prompts)} unsafe prompts to {unsafe_path}")
    print(f"Safe   prompt chars  — min {min(safe_lens)}, max {max(safe_lens)}, mean {sum(safe_lens)/len(safe_lens):.0f}")
    print(f"Unsafe prompt chars  — min {min(unsafe_lens)}, max {max(unsafe_lens)}, mean {sum(unsafe_lens)/len(unsafe_lens):.0f}")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()
