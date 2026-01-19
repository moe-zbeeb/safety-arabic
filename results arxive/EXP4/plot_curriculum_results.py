import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/zbibm/Safety-Arabic/benchmarking-scripts/curriculum_results.json', 'r') as f:
    data = json.load(f)

# Organize data by model
models = {}
for entry in data:
    model = entry['model']
    if model not in models:
        models[model] = {
            'stages': [],
            'safe_refusal': [],
            'unsafe_refusal': [],
            'base_safe': entry['baseSafeRefusal'],
            'base_unsafe': entry['baseUnsafeRefusal']
        }
    models[model]['stages'].append(entry['stage'])
    models[model]['safe_refusal'].append(entry['safe_refusal_rate'])
    models[model]['unsafe_refusal'].append(entry['unsafe_refusal_rate'])

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

colors = {
    'safe': '#e74c3c',      # Red for safe refusal (we want this LOW)
    'unsafe': '#27ae60',    # Green for unsafe refusal (we want this HIGH)
    'base_safe': '#c0392b',
    'base_unsafe': '#1e8449'
}

model_names = list(models.keys())

for idx, model in enumerate(model_names):
    ax = axes[idx]
    model_data = models[model]

    # Add base as first point
    stages = ['Base'] + model_data['stages']
    safe_rates = [model_data['base_safe']] + model_data['safe_refusal']
    unsafe_rates = [model_data['base_unsafe']] + model_data['unsafe_refusal']

    x = np.arange(len(stages))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, safe_rates, width, label='Safe Refusal %', color=colors['safe'], alpha=0.8)
    bars2 = ax.bar(x + width/2, unsafe_rates, width, label='Unsafe Refusal %', color=colors['unsafe'], alpha=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

    # Highlight base with different edge
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(2)
    bars2[0].set_edgecolor('black')
    bars2[0].set_linewidth(2)

    ax.set_ylabel('Refusal Rate (%)')
    ax.set_title(model, fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)

# Hide the 6th subplot (we only have 5 models)
axes[5].axis('off')

# Add summary text in the 6th subplot area
summary_text = """
Key Insights:

• Safe Refusal (Red): Lower is better
  Models should NOT refuse safe queries

• Unsafe Refusal (Green): Higher is better
  Models SHOULD refuse unsafe queries

• Base: Original model performance
  (highlighted with black border)

• Best stage for most models: 95ben_5ref
  Lowest safe refusal while maintaining
  >90% unsafe refusal

• Fanar-1-9B: Best at 75ben_25ref
  (highest unsafe refusal rate)
"""
axes[5].text(0.1, 0.5, summary_text, transform=axes[5].transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Curriculum Learning: Safe vs Unsafe Refusal Rates Across Training Stages',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/zbibm/Safety-Arabic/benchmarking-scripts/curriculum_results_plot.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/zbibm/Safety-Arabic/benchmarking-scripts/curriculum_results_plot.pdf',
            bbox_inches='tight', facecolor='white')
print("Plot saved to curriculum_results_plot.png and curriculum_results_plot.pdf")
