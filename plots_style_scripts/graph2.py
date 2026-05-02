import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def set_aesthetic_style():
    """Applies the minimalist, report-style aesthetic."""
    plt.rcParams['font.family'] = 'monospace' # Gives that slight typewriter/code feel to the numbers
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = False
    # Only keep a faint bottom spine
    plt.rcParams['axes.spines.bottom'] = False 
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'y'   # <--- The corrected line!
    plt.rcParams['grid.color'] = '#f0f0f0'
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

set_aesthetic_style()

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
fig.subplots_adjust(bottom=0.18) # Leave room for the custom bottom banner

# Exact data points approximated from your reference image
x_eval = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
y_pass1 = np.array([46.0, 46.2, 47.3, 48.1, 47.4, 46.9, 43.6, 40.8, 36.6, 30.9])
y_pass5 = np.array([61.6, 61.6, 62.4, 64.0, 63.8, 64.0, 60.5, 58.4, 54.3, 54.1])

# Colors
C_PASS1 = '#4c78a8' # Slate blue
C_PASS5 = '#b55d3e' # Rust / Terracotta
C_HIGHLIGHT = '#cfa244' # Golden

# 1. Plot the dashed baselines
ax.axhline(42.4, color=C_PASS1, linestyle='--', linewidth=1.5, alpha=0.3)
ax.axhline(53.5, color=C_PASS5, linestyle='--', linewidth=1.5, alpha=0.3)

# 2. Plot the main data lines with dot markers
ax.plot(x_eval, y_pass1, marker='o', markersize=6, linewidth=2.5, color=C_PASS1, label='pass@1')
ax.plot(x_eval, y_pass5, marker='o', markersize=6, linewidth=2.5, color=C_PASS5, label='pass@5')

# 3. Add the Golden Highlight Marker at (0.9, 48.1)
highlight_idx = 3 # index for 0.9
ax.plot(x_eval[highlight_idx], y_pass1[highlight_idx], marker='o', 
        markersize=12, color=C_HIGHLIGHT, zorder=5)

# Add the text label "48.1%" exactly above the golden marker
ax.annotate('48.1%', 
            xy=(x_eval[highlight_idx], y_pass1[highlight_idx]), 
            xytext=(0, 12), textcoords='offset points',
            ha='center', va='bottom', color=C_PASS1, fontsize=14)

# 4. Format Axes
ax.set_ylim(25, 75)
ax.set_yticks([30, 40, 50, 60, 70])
ax.set_xticks(x_eval)

# Style tick labels
ax.tick_params(axis='both', length=0, pad=10, colors='#888888', labelsize=12)
ax.set_xlabel('$T\_eval$', fontsize=12, color='#888888', style='italic')
ax.set_ylabel('Accuracy (%)', fontsize=12, color='#888888', style='italic', labelpad=15)

# 5. Legend
ax.legend(loc='upper right', frameon=False, ncol=2, 
          labelcolor='#666666', fontsize=11, bbox_to_anchor=(0.95, 1.05))

# 6. Custom Gray Banner at the Bottom
# Draw a full-width light gray rectangle across the bottom of the figure
rect = patches.Rectangle((0.02, 0.02), 0.96, 0.08, transform=fig.transFigure, 
                         facecolor='#f4f5f7', edgecolor='none', zorder=-1, clip_on=False)
fig.add_artist(rect)

# Draw the 'b' circle inside the banner
fig.text(0.40, 0.06, " b ", transform=fig.transFigure, ha="center", va="center",
         fontsize=10, fontweight='bold', color='#444444',
         bbox=dict(boxstyle="circle,pad=0.2", facecolor="#e5e7eb", edgecolor="none"))

# Draw the banner text
fig.text(0.50, 0.06, "Best accuracy vs $T_{eval}$", transform=fig.transFigure, 
         ha="center", va="center", fontsize=12, fontweight='bold', color='#777777')

plt.savefig('reproduced_accuracy_plot.png', dpi=300, bbox_inches='tight')
print("Graph saved successfully as reproduced_accuracy_plot.png")
plt.show()