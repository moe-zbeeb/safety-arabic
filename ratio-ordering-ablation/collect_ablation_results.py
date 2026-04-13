#!/usr/bin/env python3
"""
Collect all ratio × ordering ablation results into a single CSV and JSON.

Walks results_ratio_ordering_ablation/<model>/<ratio>__<ordering>_summary.json
and produces:
  - results_ratio_ordering_ablation/combined_results.csv
  - results_ratio_ordering_ablation/combined_results.json
  - a printed summary table on stdout
"""

import json
import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_ratio_ordering_ablation"
OUTPUT_CSV = RESULTS_DIR / "combined_results.csv"
OUTPUT_JSON = RESULTS_DIR / "combined_results.json"

CSV_FIELDS = [
    "model",
    "ratio",
    "ordering",
    "total_prompts",
    "refused_count",
    "accepted_count",
    "refusal_rate",
    "acceptance_rate",
    "safe_prompts_count",
    "unsafe_prompts_count",
    "safe_prompts_refused",
    "unsafe_prompts_refused",
    "safe_refusal_rate",
    "unsafe_refusal_rate",
]


def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: no results directory at {RESULTS_DIR}")
        sys.exit(1)

    rows = []

    for summary_file in sorted(RESULTS_DIR.rglob("*_summary.json")):
        model = summary_file.parent.name
        run_name = summary_file.stem.replace("_summary", "")

        parts = run_name.split("__", 1)
        if len(parts) != 2:
            continue
        ratio, ordering = parts

        with open(summary_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {"model": model, "ratio": ratio, "ordering": ordering}
        for key in CSV_FIELDS[3:]:
            row[key] = data.get(key, "")
        rows.append(row)

    if not rows:
        print("ERROR: no summary JSON files found under {RESULTS_DIR}")
        sys.exit(1)

    rows.sort(key=lambda r: (r["model"], r["ratio"], r["ordering"]))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Collected {len(rows)} results")
    print(f"  CSV:  {OUTPUT_CSV}")
    print(f"  JSON: {OUTPUT_JSON}")

    if rows:
        print()
        header = (
            f"{'Model':<30} {'Ratio':<15} {'Ordering':<20} "
            f"{'Refusal%':>9} {'SafeRef%':>9} {'UnsafeRef%':>11}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            ref = f"{r['refusal_rate']:.1f}" if isinstance(r["refusal_rate"], (int, float)) else str(r["refusal_rate"])
            srf = f"{r['safe_refusal_rate']:.1f}" if isinstance(r["safe_refusal_rate"], (int, float)) else str(r["safe_refusal_rate"])
            urf = f"{r['unsafe_refusal_rate']:.1f}" if isinstance(r["unsafe_refusal_rate"], (int, float)) else str(r["unsafe_refusal_rate"])
            print(f"{r['model']:<30} {r['ratio']:<15} {r['ordering']:<20} {ref:>9} {srf:>9} {urf:>11}")


if __name__ == "__main__":
    main()
