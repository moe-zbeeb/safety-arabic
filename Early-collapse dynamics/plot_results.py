#!/usr/bin/env python3
"""
Plot safe/unsafe refusal rates across checkpoints for 5 models × 3 mixes.

Input layout:
    /Users/mohamad22/Desktop/EXP8/Eval/output/
      ALLaM/{90_safe_10_ref, 95_safe_05_ref, 100%_ref}/*.json
      Fanar/{...}/*.json
      Llama/{...}/*.json
      Qwen_3B/{...}/*.json
      Qwen_7B/{...}/*.json

Each JSON summary is expected to contain:
    safe_refusal_rate    (0-100)
    unsafe_refusal_rate  (0-100)
    refused_count        (int, optional — used for overall refusal_rate)
    total_prompts        (int, optional — used for overall refusal_rate)
    refusal_rate         (0-100, optional — used directly if present)

Output:
    plot_results.png — 2x5 grid (rows: safe/unsafe, columns: models)
    plot_results.csv — long-format table with refusal_rate, safe_refusal_rate,
                       unsafe_refusal_rate (all rounded to 2 decimals)
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------- CONFIG -----------------------

BASE_DIR = Path("/Users/mohamad22/Desktop/EXP8/Eval/output")
OUT_PNG  = BASE_DIR.parent / "plot_results.png"
OUT_CSV  = BASE_DIR.parent / "plot_results.csv"

# Display order and labels for the columns of the figure
MODELS = [
    ("ALLaM",   "ALLaM"),
    ("Fanar",   "Fanar"),
    ("Llama",   "Llama3-8B"),
    ("Qwen_3B", "Qwen2.5-3B"),
    ("Qwen_7B", "Qwen2.5-7B"),
]

# Maps mix-folder-name -> (display label, color)
MIXES = [
    ("100%_ref",       "100% Refusals", "#e63946"),   # red
    ("95_safe_05_ref", "95/5 Mix",      "#1d74f5"),   # blue
    ("90_safe_10_ref", "90/10 Mix",     "#2ca02c"),   # green
]

# Expected checkpoints (matches SAVE_STEPS in training)
SAVE_STEPS = [5, 10, 15, 20, 25, 30, 40, 60, 80, 100]

# ----------------------- HELPERS -----------------------

CKPT_RE = re.compile(r"checkpoint[-_](\d+)", re.IGNORECASE)
BASE_RE = re.compile(r"base", re.IGNORECASE)


def parse_step_from_filename(name):
    """Return step number from a summary filename, or 'base', or None."""
    if BASE_RE.search(name):
        return "base"
    m = CKPT_RE.search(name)
    if m:
        return int(m.group(1))
    return None


def compute_overall_refusal_rate(j):
    """
    Prefer explicit 'refusal_rate' field; otherwise compute from
    refused_count / total_prompts. Return None if neither is available.
    """
    if "refusal_rate" in j:
        try:
            return float(j["refusal_rate"])
        except (TypeError, ValueError):
            pass
    refused = j.get("refused_count")
    total   = j.get("total_prompts")
    if refused is not None and total:
        try:
            return 100.0 * float(refused) / float(total)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def load_mix_data(mix_dir):
    """
    Return {step_or_'base': {'overall': float, 'safe': float, 'unsafe': float}}
    for all summary JSONs in mix_dir. Missing/malformed files are skipped
    with a warning.
    """
    data = {}
    if not mix_dir.exists():
        return data
    for f in sorted(mix_dir.glob("*.json")):
        step = parse_step_from_filename(f.name)
        if step is None:
            print(f"  [skip unknown] {f.name}")
            continue
        try:
            with open(f) as fp:
                j = json.load(fp)
            data[step] = {
                "overall": compute_overall_refusal_rate(j),
                "safe":    float(j["safe_refusal_rate"]),
                "unsafe":  float(j["unsafe_refusal_rate"]),
            }
        except Exception as e:
            print(f"  [skip error]   {f.name}: {e}")
    return data


def series_for_mix(mix_data, metric):
    """
    Given a mix_data dict, return (x_labels, y_values) for plotting.
    Only includes steps actually present in mix_data.
    """
    xs, ys = [], []
    if "base" in mix_data and mix_data["base"].get(metric) is not None:
        xs.append("Base")
        ys.append(mix_data["base"][metric])
    for step in SAVE_STEPS:
        if step in mix_data and mix_data[step].get(metric) is not None:
            xs.append(str(step))
            ys.append(mix_data[step][metric])
    return xs, ys


# ----------------------- LOAD EVERYTHING -----------------------

print(f"Loading summaries from: {BASE_DIR}")
all_data = {}   # all_data[model_folder][mix_folder] = {step: {overall, safe, unsafe}}
csv_rows = []

for model_folder, model_label in MODELS:
    model_dir = BASE_DIR / model_folder
    all_data[model_folder] = {}
    if not model_dir.exists():
        print(f"[missing model dir] {model_dir}")
        continue
    print(f"\n{model_folder}:")
    for mix_folder, mix_label, _color in MIXES:
        mix_dir = model_dir / mix_folder
        mix_data = load_mix_data(mix_dir)
        all_data[model_folder][mix_folder] = mix_data
        print(f"  {mix_folder:20s}  {len(mix_data)} summaries")
        for step, rates in mix_data.items():
            csv_rows.append({
                "model": model_label,
                "mix":   mix_label,
                "step":  step,
                "refusal_rate":        rates["overall"],
                "safe_refusal_rate":   rates["safe"],
                "unsafe_refusal_rate": rates["unsafe"],
            })

# ----------------------- CSV -----------------------

df = pd.DataFrame(csv_rows)

# Sort for readability: model, mix, then numeric step (put 'base' first)
df["_step_order"] = df["step"].apply(lambda s: -1 if s == "base" else int(s))
df = df.sort_values(["model", "mix", "_step_order"]).drop(columns="_step_order")

# Round rate columns to 2 decimals
rate_cols = ["refusal_rate", "safe_refusal_rate", "unsafe_refusal_rate"]
df[rate_cols] = df[rate_cols].round(2)

df.to_csv(OUT_CSV, index=False)
print(f"\nWrote CSV: {OUT_CSV}  ({len(df)} rows)")

# ----------------------- PLOT -----------------------

n_cols = len(MODELS)
fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8), sharey="row")

# Ensure axes is 2D even with n_cols=1
if n_cols == 1:
    axes = axes.reshape(2, 1)

for col_idx, (model_folder, model_label) in enumerate(MODELS):
    for row_idx, metric in enumerate(["safe", "unsafe"]):
        ax = axes[row_idx][col_idx]

        # Consistent x-axis label order across all panels
        all_x = ["Base"] + [str(s) for s in SAVE_STEPS]

        for mix_folder, mix_label, color in MIXES:
            mix_data = all_data.get(model_folder, {}).get(mix_folder, {})
            if not mix_data:
                continue
            xs, ys = series_for_mix(mix_data, metric)
            # Map string labels to integer positions based on all_x
            x_pos = [all_x.index(x) for x in xs]
            ax.plot(x_pos, ys, "-o", color=color, label=mix_label,
                    markersize=5, linewidth=1.8)

        # Cosmetics
        if row_idx == 0:
            ax.set_title(model_label, fontsize=12, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(
                "Safe Refusal Rate (%)" if metric == "safe"
                else "Unsafe Refusal Rate (%)",
                fontsize=10,
            )
        ax.set_xticks(range(len(all_x)))
        ax.set_xticklabels(all_x, rotation=45, fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(0, 105)

# One shared legend at the top
handles, labels = axes[0][0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=len(MIXES),
               bbox_to_anchor=(0.5, 1.02), fontsize=11, frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Wrote plot: {OUT_PNG}")
