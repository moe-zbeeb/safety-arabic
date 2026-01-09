#!/usr/bin/env python3
"""
Histogram visualization for baseline (untrained) model refusal rates.
Shows original model performance before safety training.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Read baseline results
baseline_dir = '/Users/mohammadzbeeb/Safety-Arabic/results arxive/Exp2'

models_data = []
files = [
    'Fanar-1-9B_summary.json',
    'Meta-Llama-3-8B-Instruct_summary.json',
    'Qwen2.5-3B-Instruct_summary.json',
    'Qwen2.5-7B-Instruct_summary.json'
]

for file in files:
    with open(f'{baseline_dir}/{file}', 'r') as f:
        data = json.load(f)
        models_data.append(data)

# Extract data
models = [d['model'].replace('-Instruct', '').replace('Meta-Llama-3-8B', 'Llama-3-8B') for d in models_data]
safe_refusal = [d['safe_refusal_rate'] for d in models_data]
unsafe_refusal = [d['unsafe_refusal_rate'] for d in models_data]

# Create histogram
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.35

# Create bars with colors
bars1 = ax.bar(x - width/2, safe_refusal, width,
              label='Safe Refusal Rate (False Positives)',
              color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=1.5)

bars2 = ax.bar(x + width/2, unsafe_refusal, width,
              label='Unsafe Refusal Rate (True Positives)',
              color='#27AE60', alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom',
               fontsize=11, fontweight='bold')

add_labels(bars1)
add_labels(bars2)

# Customize plot
ax.set_ylabel('Refusal Rate (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_title('Baseline Refusal Rates: Original Models Without Safety Training',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

# Add grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
ax.set_axisbelow(True)

# Set y-axis limit
ax.set_ylim(0, 85)

# Add reference line at ideal unsafe refusal rate
ax.axhline(y=90, color='green', linestyle=':', linewidth=2, alpha=0.4, label='Target: 90%+ Unsafe Refusal')
ax.axhline(y=10, color='red', linestyle=':', linewidth=2, alpha=0.4, label='Target: <10% Safe Refusal')

# Update legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right', fontsize=11, framealpha=0.95)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()

# Save
output_path = '/Users/mohammadzbeeb/Safety-Arabic/results arxive/Exp2/baseline_refusal_histogram.png'
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
print(f"Saved baseline histogram to: {output_path}")

# Print summary statistics
print("\nBaseline Model Performance (No Training):")
print("=" * 60)
for i, model in enumerate(models):
    print(f"\n{model}:")
    print(f"  Safe Refusal:   {safe_refusal[i]:5.1f}% (False Positives)")
    print(f"  Unsafe Refusal: {unsafe_refusal[i]:5.1f}% (True Positives)")
    print(f"  Gap to Target:  Safe={safe_refusal[i]-10:.1f}%, Unsafe={90-unsafe_refusal[i]:.1f}%")

plt.show()
