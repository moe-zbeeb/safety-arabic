#!/usr/bin/env python3
"""
Visualization script for Safety-Arabic EXP3 refusal rates analysis.

This script parses the refusal rates markdown table and creates comprehensive
visualizations comparing models trained on:
1. 100% refusals data
2. 70% general + 30% refusals (mix) data

The plots show safe refusal rates (false positives) vs unsafe refusal rates
(true positives) across different checkpoints for each model.
"""

import re
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------------------------
# Read markdown file
# ---------------------------
with open('/Users/mohammadzbeeb/Safety-Arabic/results arxive/EXP3/refusal_rates_table.md', 'r') as f:
    md_text = f.read()

# ---------------------------
# Parse markdown table into tidy dataframe
# Columns: model, checkpoint, data_type, safe_refusal, unsafe_refusal
# ---------------------------
def _parse_md_table(table_md: str) -> pd.DataFrame:
    """
    Parse a single markdown table into a dataframe.
    Handles multi-row headers with subheaders.

    Args:
        table_md: String containing markdown table

    Returns:
        DataFrame with columns: checkpoint, data_type, safe_refusal, unsafe_refusal
    """
    lines = [l.strip() for l in table_md.strip().splitlines() if l.strip()]
    if len(lines) < 4:
        return pd.DataFrame()

    def split_row(row: str):
        """Split table row by pipe delimiter"""
        row = row.strip().strip("|")
        return [c.strip() for c in row.split("|")]

    # Find the subheader row (contains "Safe Refusal %" and "Unsafe Refusal %")
    subheader_idx = None
    for i, line in enumerate(lines):
        if "Safe Refusal %" in line and "Unsafe Refusal %" in line:
            subheader_idx = i
            break

    if subheader_idx is None:
        return pd.DataFrame()

    # Get column names from subheader row
    subheader = split_row(lines[subheader_idx])

    # Data rows start after subheader
    data_start_idx = subheader_idx + 1

    # Find separator row after subheader (if exists)
    if data_start_idx < len(lines) and re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", lines[data_start_idx]):
        data_start_idx += 1

    # Parse data rows
    data = []
    for r in lines[data_start_idx:]:
        # Skip separator rows
        if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", r):
            continue
        cols = split_row(r)
        if len(cols) == len(subheader):
            data.append(cols)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=subheader)

    # Find checkpoint column (usually first column, might be empty string name)
    ck_col = df.columns[0]

    # Find refusal columns (in order: safe_refusals, unsafe_refusals, safe_mix, unsafe_mix)
    # Columns should be: [checkpoint, Safe Refusal %, Unsafe Refusal %, Safe Refusal %, Unsafe Refusal %]
    if len(df.columns) != 5:
        return pd.DataFrame()

    # Build tidy format: one row per (checkpoint, data_type) combination
    out = []
    for _, row in df.iterrows():
        ck = str(row[ck_col]).strip()
        if not ck:  # Skip empty checkpoint rows
            continue

        # Refusals data (columns 1 and 2)
        out.append({
            "checkpoint": ck,
            "data_type": "refusals",
            "safe_refusal": row.iloc[1],
            "unsafe_refusal": row.iloc[2],
        })

        # Mix data (columns 3 and 4)
        out.append({
            "checkpoint": ck,
            "data_type": "mix",
            "safe_refusal": row.iloc[3],
            "unsafe_refusal": row.iloc[4],
        })

    tidy = pd.DataFrame(out)

    # Convert refusal rates to numeric (remove % signs)
    for c in ["safe_refusal", "unsafe_refusal"]:
        tidy[c] = (
            tidy[c]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        tidy[c] = pd.to_numeric(tidy[c], errors="coerce")

    return tidy

def parse_markdown(md: str) -> pd.DataFrame:
    """
    Parse the full markdown document containing multiple model tables.

    Args:
        md: Full markdown text with multiple model sections

    Returns:
        DataFrame with all models' data
    """
    # Split by model headings like: "## FANAR Model"
    # Capture model name and the table that follows
    pattern = re.compile(r"##\s+(.+?)\s+Model\s*(.*?)(?=\n##\s+.+?\s+Model|\Z)", re.S)
    matches = pattern.findall(md)

    all_parts = []
    for model_name, block in matches:
        # Extract the first markdown table inside the block
        table_match = re.search(r"(\|.*\|\s*\n\|.*\|\s*\n(?:\|.*\|\s*\n?)*)", block, re.S)
        if not table_match:
            continue
        table_md = table_match.group(1)
        df = _parse_md_table(table_md)
        if df.empty:
            continue
        df["model"] = model_name.strip()
        all_parts.append(df)

    if not all_parts:
        return pd.DataFrame()

    out = pd.concat(all_parts, ignore_index=True)

    # Order checkpoints by step number when possible
    def ck_key(x: str):
        """Generate sort key for checkpoint names"""
        x = str(x)
        if x == "final-model":
            return (10**12, 0)  # Put final-model at the end
        m = re.search(r"checkpoint-(\d+)", x)
        if m:
            return (int(m.group(1)), 0)
        return (10**11, 1)

    out["ck_order"] = out["checkpoint"].map(ck_key)
    out = out.sort_values(["model", "data_type", "ck_order"]).drop(columns=["ck_order"])

    return out

# Parse the markdown file
df = parse_markdown(md_text)
if df.empty:
    raise ValueError("Parsing produced no rows. Make sure md_text contains your model sections + tables.")

print(f"Loaded {len(df)} rows for {df['model'].nunique()} models")
print(f"Models: {df['model'].unique().tolist()}")

# ---------------------------
# Plot 1: Grid plot for each data type
# 2 rows (safe/unsafe) × N columns (models)
# ---------------------------
def plot_refusals_grid(df: pd.DataFrame, data_type: str, suptitle: str = None):
    """
    Create a grid plot showing safe and unsafe refusal rates across checkpoints.

    Args:
        df: DataFrame with refusal data
        data_type: Either 'refusals' or 'mix'
        suptitle: Title for the entire figure

    Returns:
        Figure object
    """
    sub = df[df["data_type"] == data_type].copy()
    models = list(dict.fromkeys(sub["model"].tolist()))  # Preserve order

    ncols = len(models)
    nrows = 2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 8.0),
        sharex=False,
        sharey="row",
        constrained_layout=True,
    )

    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    metric_rows = [
        ("safe_refusal", "Safe Refusal (%) - False Positives"),
        ("unsafe_refusal", "Unsafe Refusal (%) - True Positives"),
    ]

    # Color scheme for better visualization
    colors = ['#2E86AB', '#A23B72']

    for j, model in enumerate(models):
        mdf = sub[sub["model"] == model].copy()

        # Extract checkpoint numbers for x-axis
        x_labels = list(mdf["checkpoint"].astype(str))
        x_pos = range(len(x_labels))

        for i, (metric, ylab) in enumerate(metric_rows):
            ax = axes[i, j]
            y = mdf[metric].astype(float).tolist()

            # Plot with markers and lines
            ax.plot(x_pos, y, marker='o', linewidth=2.5,
                   markersize=8, color=colors[i], alpha=0.8)

            # Add title and labels
            ax.set_title(model, fontsize=13, fontweight='bold')
            if j == 0:
                ax.set_ylabel(ylab, fontsize=11, fontweight='bold')
            ax.set_xlabel("Checkpoint", fontsize=10)

            # Set x-axis labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

            # Add grid for readability
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

            # Set y-axis limits based on metric
            if metric == "safe_refusal":
                # For safe refusal, focus on lower range for mix data
                if data_type == "mix":
                    ax.set_ylim(-5, max(y) + 10)
                else:
                    ax.set_ylim(95, 101)
            else:
                # For unsafe refusal, always show full range
                ax.set_ylim(80, 101)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)

    return fig

# Generate grid plots
print("\nGenerating grid plots...")
fig1 = plot_refusals_grid(df, "refusals", suptitle="Training on 100% Refusals Data")
fig2 = plot_refusals_grid(df, "mix", suptitle="Training on Mix Data (70% General + 30% Refusals)")

# ---------------------------
# Plot 2: Scatter plot showing trade-off between safe and unsafe refusal rates
# ---------------------------
def plot_tradeoff_scatter(df: pd.DataFrame):
    """
    Create scatter plot showing the trade-off between safe and unsafe refusal rates.

    Args:
        df: DataFrame with refusal data

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    data_types = ["refusals", "mix"]
    titles = ["100% Refusals Data", "Mix Data (70% General + 30% Refusals)"]

    for idx, (data_type, title) in enumerate(zip(data_types, titles)):
        ax = axes[idx]
        sub = df[df["data_type"] == data_type].copy()

        # Get unique models
        models = sub["model"].unique()
        colors_map = dict(zip(models, sns.color_palette("husl", len(models))))

        # Plot each model with different color and marker
        for model in models:
            mdf = sub[sub["model"] == model]
            ax.scatter(mdf["safe_refusal"], mdf["unsafe_refusal"],
                      s=100, alpha=0.7, label=model,
                      color=colors_map[model], edgecolors='black', linewidth=1)

        # Add ideal region shading (low safe refusal, high unsafe refusal)
        ax.axhspan(90, 100, xmin=0, xmax=0.2, alpha=0.1, color='green',
                  label='Ideal Region')

        ax.set_xlabel("Safe Refusal Rate (%) - False Positives",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel("Unsafe Refusal Rate (%) - True Positives",
                     fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.4, linestyle='--')

        # Set appropriate limits
        if data_type == "refusals":
            ax.set_xlim(95, 101)
        else:
            ax.set_xlim(0, 50)
        ax.set_ylim(80, 101)

    fig.suptitle("Safety-Utility Trade-off: Safe vs Unsafe Refusal Rates",
                fontsize=16, fontweight='bold', y=0.995)

    return fig

print("Generating trade-off scatter plot...")
fig3 = plot_tradeoff_scatter(df)

# ---------------------------
# Plot 3: Final checkpoint comparison bar chart
# ---------------------------
def plot_final_checkpoint_comparison(df: pd.DataFrame):
    """
    Create bar chart comparing final checkpoint performance across models.

    Args:
        df: DataFrame with refusal data

    Returns:
        Figure object
    """
    # Get final checkpoint (checkpoint-313 or final-model)
    final_df = df[df["checkpoint"].str.contains("313|final-model")].copy()

    # For each model, take the last checkpoint
    final_df = final_df.groupby(["model", "data_type"]).last().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    metrics = ["safe_refusal", "unsafe_refusal"]
    data_types = ["refusals", "mix"]
    metric_labels = ["Safe Refusal Rate (%)", "Unsafe Refusal Rate (%)"]
    type_labels = ["100% Refusals Data", "Mix Data"]

    for i, metric in enumerate(metrics):
        for j, data_type in enumerate(data_types):
            ax = axes[i, j]
            sub = final_df[final_df["data_type"] == data_type].copy()

            models = sub["model"].tolist()
            values = sub[metric].tolist()

            # Create bars with gradient colors
            bars = ax.bar(models, values, width=0.6,
                         color=sns.color_palette("viridis", len(models)),
                         edgecolor='black', linewidth=1.5, alpha=0.8)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}%', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')

            ax.set_ylabel(metric_labels[i], fontsize=11, fontweight='bold')
            ax.set_title(f"{metric_labels[i]} - {type_labels[j]}",
                        fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.grid(True, axis='y', alpha=0.4, linestyle='--')

            # Set y-axis limits
            if metric == "safe_refusal":
                if data_type == "refusals":
                    ax.set_ylim(95, 101)
                else:
                    ax.set_ylim(0, max(values) + 10)
            else:
                ax.set_ylim(80, 101)

    fig.suptitle("Final Checkpoint Performance Comparison",
                fontsize=16, fontweight='bold', y=0.995)

    return fig

print("Generating final checkpoint comparison...")
fig4 = plot_final_checkpoint_comparison(df)

# ---------------------------
# Plot 4: Line plot comparing refusals vs mix for each model
# ---------------------------
def plot_refusals_vs_mix_comparison(df: pd.DataFrame):
    """
    Create line plots comparing refusals vs mix training for each model.

    Args:
        df: DataFrame with refusal data

    Returns:
        Figure object
    """
    models = df["model"].unique()
    ncols = len(models)

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10),
                            sharex=False, constrained_layout=True)

    if ncols == 1:
        axes = axes.reshape(2, 1)

    metrics = ["safe_refusal", "unsafe_refusal"]
    metric_labels = ["Safe Refusal (%) - Lower is Better",
                    "Unsafe Refusal (%) - Higher is Better"]

    for j, model in enumerate(models):
        mdf = df[df["model"] == model].copy()

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]

            # Plot refusals data
            ref_data = mdf[mdf["data_type"] == "refusals"]
            x_ref = range(len(ref_data))
            y_ref = ref_data[metric].tolist()

            # Plot mix data
            mix_data = mdf[mdf["data_type"] == "mix"]
            x_mix = range(len(mix_data))
            y_mix = mix_data[metric].tolist()

            ax.plot(x_ref, y_ref, marker='s', linewidth=2.5, markersize=7,
                   label='100% Refusals', color='#E63946', alpha=0.8)
            ax.plot(x_mix, y_mix, marker='o', linewidth=2.5, markersize=7,
                   label='Mix (70-30)', color='#06A77D', alpha=0.8)

            ax.set_title(model, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold')
            ax.set_xlabel("Checkpoint", fontsize=9)

            # Set x-axis labels
            x_labels = ref_data["checkpoint"].tolist()
            ax.set_xticks(x_ref)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

            # Adjust y-axis based on data
            if metric == "safe_refusal":
                y_min = min(min(y_ref), min(y_mix)) - 5
                y_max = max(max(y_ref), max(y_mix)) + 5
                ax.set_ylim(max(0, y_min), min(105, y_max))

    fig.suptitle("Refusals vs Mix Training Comparison",
                fontsize=16, fontweight='bold', y=0.995)

    return fig

print("Generating refusals vs mix comparison...")
fig5 = plot_refusals_vs_mix_comparison(df)

# ---------------------------
# Save all figures
# ---------------------------
print("\nSaving figures...")
output_dir = '/Users/mohammadzbeeb/Safety-Arabic/results arxive/EXP3'

fig1.savefig(f"{output_dir}/plot1_refusals_grid.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"{output_dir}/plot2_mix_grid.png", dpi=300, bbox_inches="tight")
fig3.savefig(f"{output_dir}/plot3_tradeoff_scatter.png", dpi=300, bbox_inches="tight")
fig4.savefig(f"{output_dir}/plot4_final_comparison.png", dpi=300, bbox_inches="tight")
fig5.savefig(f"{output_dir}/plot5_refusals_vs_mix.png", dpi=300, bbox_inches="tight")

print("All plots saved successfully!")
print(f"Output directory: {output_dir}")

# Display plots
plt.show()
