"""
plot_safety_results.py

Standalone visualization script for safety benchmark summary files.
Reads JSON files matching '*_summary.json' from an input directory and
produces comparison plots across models and dialects.

Usage:
    python plot_safety_results.py --input Results --Results plots
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------- config ----------
DIALECT_ORDER = ["MSA", "EGY", "LEV", "ARABIZI", "NOISY"]
FILENAME_RE = re.compile(r"^(?P<model>.+?)_Arabic[ _][Bb]oundary-set_(?P<dialect>[A-Z]+)_summary\.json$")


# ---------- data loading ----------
def load_summaries(input_dir: Path):
    """Return list of dicts: {model, dialect, refusal_rate, safe_refusal_rate, unsafe_refusal_rate, per_category}."""
    records = []
    for path in sorted(input_dir.glob("*_summary.json")):
        match = FILENAME_RE.match(path.name)
        if not match:
            print(f"[skip] cannot parse filename: {path.name}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records.append({
            "model": match.group("model"),
            "dialect": match.group("dialect"),
            "refusal_rate": float(data.get("refusal_rate", 0.0)),
            "safe_refusal_rate": float(data.get("safe_refusal_rate", 0.0)),
            "unsafe_refusal_rate": float(data.get("unsafe_refusal_rate", 0.0)),
            "total": int(data.get("total_prompts", 0)),
            "per_category": data.get("per_category", {}),
        })
    return records


def organize(records):
    """Return (models_sorted, dialects_sorted, matrix_dict) where matrix_dict[metric] is 2D np.ndarray[model, dialect]."""
    models = sorted({r["model"] for r in records})
    found_dialects = {r["dialect"] for r in records}
    dialects = [d for d in DIALECT_ORDER if d in found_dialects] + sorted(found_dialects - set(DIALECT_ORDER))

    metrics = ["refusal_rate", "safe_refusal_rate", "unsafe_refusal_rate"]
    matrices = {m: np.full((len(models), len(dialects)), np.nan) for m in metrics}

    for r in records:
        i = models.index(r["model"])
        j = dialects.index(r["dialect"])
        for m in metrics:
            matrices[m][i, j] = r[m]

    return models, dialects, matrices


# ---------- plots ----------
def grouped_bar(models, dialects, matrix, title, ylabel, out_path, cmap_name="tab10"):
    fig, ax = plt.subplots(figsize=(max(12, 1.6 * len(dialects) * len(models) / 4), 6))
    n_models = len(models)
    bar_w = 0.8 / n_models
    x = np.arange(len(dialects))
    cmap = plt.get_cmap(cmap_name)

    for i, model in enumerate(models):
        offsets = x + (i - (n_models - 1) / 2) * bar_w
        ax.bar(offsets, matrix[i], width=bar_w, label=model, color=cmap(i % cmap.N))

    ax.set_xticks(x)
    ax.set_xticklabels(dialects)
    ax.set_xlabel("Dialect")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def heatmap(models, dialects, matrix, title, out_path, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=(1.4 * len(dialects) + 4, 0.6 * len(models) + 3))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(dialects)))
    ax.set_xticklabels(dialects)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Refusal rate (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def scatter_tradeoff(records, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    models = sorted({r["model"] for r in records})
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for i, model in enumerate(models):
        sub = [r for r in records if r["model"] == model]
        xs = [r["safe_refusal_rate"] for r in sub]
        ys = [r["unsafe_refusal_rate"] for r in sub]
        ax.scatter(xs, ys, s=180, alpha=0.8, marker=markers[i % len(markers)],
                   color=cmap(i % cmap.N), label=model, edgecolors="black", linewidths=0.5)
        for r in sub:
            ax.annotate(r["dialect"], (r["safe_refusal_rate"], r["unsafe_refusal_rate"]),
                        fontsize=8, xytext=(6, 6), textcoords="offset points")

    ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.4, label="y = x")
    ax.set_xlabel("Over-refusal: safe prompts refused (%) — lower is better")
    ax.set_ylabel("Correct refusal: unsafe prompts refused (%) — higher is better")
    ax.set_title("Safety vs. Helpfulness Trade-off")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def per_category_heatmap(records, out_path):
    cat_data = {}  # (model, category) -> list of refusal_rates across dialects
    for r in records:
        for cat, stats in r.get("per_category", {}).items():
            cat_data.setdefault((r["model"], cat), []).append(stats.get("refusal_rate", 0))

    if not cat_data:
        print("[skip] no per_category data found")
        return

    models = sorted({m for (m, _) in cat_data})
    cats = sorted({c for (_, c) in cat_data})
    matrix = np.full((len(models), len(cats)), np.nan)
    for (m, c), vals in cat_data.items():
        matrix[models.index(m), cats.index(c)] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(cats) + 4), 0.6 * len(models) + 3))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Per-Category Refusal Rate (averaged across dialects)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 55 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Mean refusal rate (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def write_csv(records, out_path):
    headers = ["model", "dialect", "total", "refusal_rate", "safe_refusal_rate", "unsafe_refusal_rate"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in sorted(records, key=lambda x: (x["model"], x["dialect"])):
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    print(f"[saved] {out_path}")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Visualize safety benchmark summaries.")
    parser.add_argument("--input", "-i", default="Results", help="Directory with *_summary.json files")
    parser.add_argument("--Results", "-o", default="plots", help="Directory to save plots into")
    args = parser.parse_args()

    input_dir = Path(args.input)
    Results_dir = Path(args.Results)
    Results_dir.mkdir(parents=True, exist_ok=True)

    records = load_summaries(input_dir)
    if not records:
        print(f"No summary files found in {input_dir}")
        return
    print(f"Loaded {len(records)} summary files.")

    models, dialects, matrices = organize(records)

    grouped_bar(models, dialects, matrices["refusal_rate"],
                "Overall Refusal Rate by Model & Dialect", "Refusal rate (%)",
                Results_dir / "01_overall_refusal.png")

    grouped_bar(models, dialects, matrices["safe_refusal_rate"],
                "Over-Refusal: Safe Prompts Refused (lower is better)",
                "Safe-prompt refusal (%)", Results_dir / "02_over_refusal.png", cmap_name="Reds")

    grouped_bar(models, dialects, matrices["unsafe_refusal_rate"],
                "Correct Refusal: Unsafe Prompts Refused (higher is better)",
                "Unsafe-prompt refusal (%)", Results_dir / "03_unsafe_refusal.png", cmap_name="Greens")

    heatmap(models, dialects, matrices["refusal_rate"],
            "Overall Refusal Rate Heatmap", Results_dir / "04_heatmap_overall.png")

    scatter_tradeoff(records, Results_dir / "05_safety_vs_helpfulness.png")

    per_category_heatmap(records, Results_dir / "06_per_category.png")

    write_csv(records, Results_dir / "summary_table.csv")
    print("Done.")


if __name__ == "__main__":
    main()


"""
plot_safe_vs_unsafe.py — recreate the EXP2-style bar chart for boundary-set results.

Usage:
    python plot_safe_vs_unsafe.py --input Results --Results safe_vs_unsafe.png
//'''

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

FILENAME_RE = re.compile(r"^(?P<model>.+?)_Arabic[ _][Bb]oundary-set_(?P<dialect>[A-Z]+)_summary\.json$")

# Display order + nicer labels
MODEL_ORDER = [
    "Fanar-1-9B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "ALLaM-7B-Instruct-preview",
]
DISPLAY_NAMES = {
    "Fanar-1-9B-Instruct": "Fanar-1-9B",
    "Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "ALLaM-7B-Instruct-preview": "ALLaM-7B-Instruct",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="Results")
    ap.add_argument("--Results", "-o", default="safe_vs_unsafe.png")
    args = ap.parse_args()

    safe = defaultdict(list)
    unsafe = defaultdict(list)
    for path in sorted(Path(args.input).glob("*_summary.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        d = json.load(open(path, encoding="utf-8"))
        safe[m.group("model")].append(d.get("safe_refusal_rate", 0))
        unsafe[m.group("model")].append(d.get("unsafe_refusal_rate", 0))

    # keep only models present, in MODEL_ORDER first then any extras
    models = [m for m in MODEL_ORDER if m in safe] + [m for m in safe if m not in MODEL_ORDER]
    safe_means = [np.mean(safe[m]) for m in models]
    unsafe_means = [np.mean(unsafe[m]) for m in models]
    labels = [DISPLAY_NAMES.get(m, m) for m in models]

    # plot
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 7))
    b1 = ax.bar(x - w/2, safe_means,   w, label="Safe Refusal",   color="#E0563F")
    b2 = ax.bar(x + w/2, unsafe_means, w, label="Unsafe Refusal", color="#3A8FCF")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Refusal Rate (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.Results, dpi=200, bbox_inches="tight")
    print(f"saved {args.Results}")


if __name__ == "__main__":
    main()
"""