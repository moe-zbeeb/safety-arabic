#!/usr/bin/env python3
"""
Build the 4-setup comparison table used in the project writeup:

1. base model only
2. base model + external guard
3. SFT-aligned model only
4. hybrid: SFT-aligned model + external guard

The guarded rows are read from EXP10-style summary JSON files.
The unguarded baseline and SFT-only reference numbers are the project values
used in the report/PDF comparison.
"""
import argparse
import csv
import json
from pathlib import Path


BASE_ONLY = {
    "allam": (19.36, 86.20),
    "fanar": (28.78, 74.72),
    "llama": (10.55, 79.59),
    "qwen3b": (9.52, 71.13),
    "qwen7b": (8.26, 69.86),
}

SFT_ONLY = {
    "allam": (16.50, 92.30),
    "fanar": (14.00, 90.00),
    "llama": (22.00, 92.80),
    "qwen3b": (22.60, 92.80),
    "qwen7b": (18.81, 91.95),
}

BASE_GUARD_FILES = {
    "allam": "allam_base__base_guard_response_summary.json",
    "fanar": "fanar_base__base_guard_response_summary.json",
    "llama": "llama_base__base_guard_response_summary.json",
    "qwen3b": "qwen3b_base__base_guard_response_summary.json",
    "qwen7b": "qwen7b_base__base_guard_response_summary.json",
}

HYBRID_FILES = {
    "allam": "allam_sft__sft_guard_response_summary.json",
    "fanar": "fanar_sft__sft_guard_response_summary.json",
    "llama": "llama_sft__sft_guard_response_summary.json",
    "qwen3b": "qwen3b_sft__sft_guard_response_summary.json",
    "qwen7b": "qwen7b_sft__sft_guard_response_summary.json",
}

MODEL_ORDER = ["allam", "fanar", "llama", "qwen3b", "qwen7b"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp-dir",
        default="results arxive/EXP10",
        help="Directory containing the EXP10-style *_summary.json files.",
    )
    p.add_argument(
        "--title",
        default="Four-setup comparison",
        help="Markdown title for the output table.",
    )
    return p.parse_args()


def load_summary_pair(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return round(float(data["safe_refusal_rate"]), 2), round(float(data["unsafe_refusal_rate"]), 2)


def fmt(pair):
    return f"{pair[0]:.2f} / {pair[1]:.2f}"


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    rows = []
    for model in MODEL_ORDER:
        base_guard = load_summary_pair(exp_dir / BASE_GUARD_FILES[model])
        hybrid = load_summary_pair(exp_dir / HYBRID_FILES[model])
        rows.append({
            "model": model,
            "base_only": fmt(BASE_ONLY[model]),
            "base_plus_guard": fmt(base_guard),
            "sft_only": fmt(SFT_ONLY[model]),
            "hybrid": fmt(hybrid),
        })

    csv_path = exp_dir / "four_setup_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "base_only", "base_plus_guard", "sft_only", "hybrid"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        f"# {args.title}",
        "",
        "| Model | Base only | Base + guard | SFT only | Hybrid |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| `{row['model']}` | `{row['base_only']}` | `{row['base_plus_guard']}` | `{row['sft_only']}` | `{row['hybrid']}` |"
        )

    md_path = exp_dir / "four_setup_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
