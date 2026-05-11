"""
summary_table.py

Print and save a compact summary table: Model | Variant | Safe Refusal% | Unsafe Refusal%

Usage:
    python boundary_set_robustness/summary_table.py
    python boundary_set_robustness/summary_table.py --input "results arxive/EXP6a/Results"
"""

import argparse
import json
import re
from pathlib import Path

FILENAME_RE = re.compile(r"^(?P<model>.+?)_Arabic[ _][Bb]oundary-set_(?P<dialect>[A-Z]+)_summary\.json$")
DIALECT_ORDER = ["MSA", "EGY", "LEV", "ARABIZI", "NOISY"]

MODEL_SHORT = {
    "ALLaM-7B-Instruct-preview":  "ALLaM-7B",
    "Fanar-1-9B-Instruct":        "Fanar-9B",
    "Meta-Llama-3-8B-Instruct":   "Llama3-8B",
    "Qwen2.5-3B-Instruct":        "Qwen2.5-3B",
    "Qwen2.5-7B-Instruct":        "Qwen2.5-7B",
}


def load(input_dir):
    rows = []
    for path in sorted(Path(input_dir).glob("*_summary.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        d = json.load(open(path, encoding="utf-8"))
        rows.append({
            "model":   m.group("model"),
            "dialect": m.group("dialect"),
            "safe":    d.get("safe_refusal_rate", 0.0),
            "unsafe":  d.get("unsafe_refusal_rate", 0.0),
        })
    rows.sort(key=lambda r: (r["model"], DIALECT_ORDER.index(r["dialect"]) if r["dialect"] in DIALECT_ORDER else 99))
    return rows


def print_table(rows):
    col1 = max(len(MODEL_SHORT.get(r["model"], r["model"])) for r in rows)
    header = f"{'Model':<{col1}}  {'Variant':<8}  {'Safe Refusal%':>14}  {'Unsafe Refusal%':>16}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    prev_model = None
    for r in rows:
        name = MODEL_SHORT.get(r["model"], r["model"])
        if prev_model and name != prev_model:
            print()
        print(f"{name:<{col1}}  {r['dialect']:<8}  {r['safe']:>13.1f}%  {r['unsafe']:>15.1f}%")
        prev_model = name
    print(sep)


def save_csv(rows, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Model,Variant,Safe Refusal%,Unsafe Refusal%\n")
        for r in rows:
            f.write(f"{MODEL_SHORT.get(r['model'], r['model'])},{r['dialect']},{r['safe']:.1f},{r['unsafe']:.1f}\n")
    print(f"saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="results arxive/EXP6a/Results")
    args = ap.parse_args()

    rows = load(args.input)
    if not rows:
        print("No result files found.")
        return

    print_table(rows)
    out = Path(args.input).parent / "summary_table.csv"
    save_csv(rows, out)


if __name__ == "__main__":
    main()
