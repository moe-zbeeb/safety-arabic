#!/usr/bin/env python3
"""
Build CSV tables of refusal rates from per-model AraSafe eval JSONs.

For each threshold rule (FPR<=0.01, FPR<=0.05, FPR<=0.10, Youden_J), this
writes one CSV with one row per model containing:
    - model_name
    - n_safe / n_unsafe (size of each AraSafe subpool)
    - safe_refusal_rate    = FPR  = fraction of safe prompts wrongly refused
    - unsafe_refusal_rate  = TPR  = fraction of unsafe prompts correctly refused
    - n_refused_safe
    - n_caught_unsafe
    - tau, accuracy, f1

Also writes a single combined CSV with all rules in long format.

Inputs:  inference_time_guard/eval/<model>_arasafe_metrics.json
Outputs: inference_time_guard/tables/refusal_rates__<rule>.csv
         inference_time_guard/tables/refusal_rates__all.csv
"""

import csv
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent / "eval"
OUT_DIR = Path(__file__).resolve().parent / "tables"

COLUMNS = [
    "model_name",
    "rule",
    "tau",
    "n_safe",
    "n_unsafe",
    "safe_refusal_rate",     # FPR
    "unsafe_refusal_rate",   # TPR
    "n_refused_safe",
    "n_caught_unsafe",
    "missed_unsafe",
    "accuracy",
    "f1",
    "auc_arasafe",
]


def slug(rule: str) -> str:
    return rule.replace("<=", "_le_").replace(" ", "_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(EVAL_DIR.glob("*_arasafe_metrics.json"))
    if not metrics_files:
        raise SystemExit(f"no *_arasafe_metrics.json files in {EVAL_DIR}")

    by_rule: dict[str, list[dict]] = {}
    all_rows: list[dict] = []

    for path in metrics_files:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        for tau_row in m["per_tau"]:
            row = {
                "model_name":          m["model_name"],
                "rule":                tau_row["rule"],
                "tau":                 round(tau_row["tau"], 6),
                "n_safe":              m["n_safe"],
                "n_unsafe":            m["n_unsafe"],
                "safe_refusal_rate":   round(tau_row["fpr"], 6),
                "unsafe_refusal_rate": round(tau_row["tpr"], 6),
                "n_refused_safe":      tau_row["n_refused_safe"],
                "n_caught_unsafe":     tau_row["n_caught_unsafe"],
                "missed_unsafe":       m["n_unsafe"] - tau_row["n_caught_unsafe"],
                "accuracy":            round(tau_row["accuracy"], 6),
                "f1":                  round(tau_row["f1"], 6),
                "auc_arasafe":         round(m["auc_arasafe"], 6),
            }
            by_rule.setdefault(tau_row["rule"], []).append(row)
            all_rows.append(row)

    # One CSV per rule
    for rule, rows in by_rule.items():
        out_path = OUT_DIR / f"refusal_rates__{slug(rule)}.csv"
        rows.sort(key=lambda r: r["model_name"])
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_path}  ({len(rows)} rows)")

    # Combined long-format CSV
    combined = OUT_DIR / "refusal_rates__all.csv"
    all_rows.sort(key=lambda r: (r["rule"], r["model_name"]))
    with open(combined, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {combined}  ({len(all_rows)} rows)")

    # Pretty-print to stdout, one table per rule
    for rule in sorted(by_rule):
        rows = by_rule[rule]
        print(f"\n=== {rule} ===")
        print(f"{'model':<28} {'safe_refusal':>13} {'unsafe_refusal':>15} "
              f"{'n_refused_safe':>15} {'n_caught_unsafe':>16}")
        print("-" * 90)
        for r in rows:
            print(f"{r['model_name']:<28} "
                  f"{r['safe_refusal_rate']:>13.4f} "
                  f"{r['unsafe_refusal_rate']:>15.4f} "
                  f"{r['n_refused_safe']:>15d} "
                  f"{r['n_caught_unsafe']:>16d}")


if __name__ == "__main__":
    main()
