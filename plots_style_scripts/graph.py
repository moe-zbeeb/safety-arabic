import matplotlib.pyplot as plt
import numpy as np

def set_panel_style():
    """Applies the specific styling for these subplots."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#e5e7eb' # Faint grid lines
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['axes.axisbelow'] = True
    # The axes backgrounds are slightly greyish, figure background is white
    plt.rcParams['axes.facecolor'] = '#f4f5f7' 
    plt.rcParams['figure.facecolor'] = 'white'

set_panel_style()

# Paper Colors
C_BASE = '#b55d3e' # Rust / Orange
C_SSD = '#4e74a6'  # Slate Blue

# Create a 1x4 grid of subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=120)

# Add spacing between plots
plt.subplots_adjust(wspace=0.25, bottom=0.2)

# ==========================================
# Panel A: Cumulative Mass
# ==========================================
ax = axes[0]
x_a = np.arange(1, 61)
# Simulate the "leftover mass" (1 - cumulative)
y_base_left = 10**-(1.2 + 2.4 * (x_a / 60)**0.6)
y_ssd_left = 10**-(1.5 + 4.8 * (x_a / 60)**0.4)

ax.plot(x_a, y_base_left, color=C_BASE, linewidth=2)
ax.plot(x_a, y_ssd_left, color=C_SSD, linewidth=2)

# Fill between curve and top of graph
ax.fill_between(x_a, y_ssd_left, 10**-6, color=C_SSD, alpha=0.1)

# Set Y-axis to log scale and invert it so 10^-6 is at the top
ax.set_yscale('log')
ax.invert_yaxis()

# Custom ticks to match the '1 - 10^-x' labels in the image
yticks = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
ax.set_yticks(yticks)
ax.set_yticklabels([r'$1-10^{-1}$', r'$1-10^{-2}$', r'$1-10^{-3}$', 
                    r'$1-10^{-4}$', r'$1-10^{-5}$', r'$1-10^{-6}$'], color='#555555')

ax.set_xlim(0, 60)
ax.set_xlabel('Token rank', fontsize=12, labelpad=10)
ax.set_ylabel('Cumulative mass', fontsize=12)

# ==========================================
# Panel B: Average Surviving Tokens
# ==========================================
ax = axes[1]
x_b = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
y_base_b = np.array([1.01, 1.01, 1.01, 1.02, 1.05, 1.08, 1.08, 1.08, 1.10, 1.13, 1.20, 1.28])
y_ssd_b = np.array([1.00, 1.00, 1.18, 1.33, 1.48, 1.70, 1.80, 1.98, 2.05, 2.42, 2.92, 3.20])

ax.plot(x_b, y_base_b, '.-', color=C_BASE, markersize=8, linewidth=2)
ax.plot(x_b, y_ssd_b, '.-', color=C_SSD, markersize=8, linewidth=2)

# Shade the area under the SSD curve
ax.fill_between(x_b, y_base_b, y_ssd_b, color=C_SSD, alpha=0.08)

ax.set_xticks([0.4, 0.7, 1.0, 1.4, 2.0])
ax.set_ylim(0.9, 3.4)
ax.set_xlabel(r'$T_{eval}$', fontsize=12)
ax.set_ylabel('Average surviving tokens', fontsize=12)

# ==========================================
# Panel C: Entropy after filtering
# ==========================================
ax = axes[2]
y_base_c = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.04, 0.04, 0.05, 0.07, 0.11, 0.15])
y_ssd_c = np.array([0.00, 0.00, 0.08, 0.18, 0.25, 0.37, 0.42, 0.51, 0.57, 0.71, 0.88, 0.96])

ax.plot(x_b, y_base_c, '.-', color=C_BASE, markersize=8, linewidth=2)
ax.plot(x_b, y_ssd_c, '.-', color=C_SSD, markersize=8, linewidth=2)

# Shade the area under the SSD curve
ax.fill_between(x_b, y_base_c, y_ssd_c, color=C_SSD, alpha=0.08)

ax.set_xticks([0.4, 0.7, 1.0, 1.4, 2.0])
ax.set_ylim(0.0, 1.05)
ax.set_xlabel(r'$T_{eval}$', fontsize=12)
ax.set_ylabel('Entropy after filtering (nats)', fontsize=12)

# ==========================================
# Panel D: Bar Chart
# ==========================================
ax = axes[3]
labels = ['99.9%', '99.5%', '99.0%', '98.0%', '95.0%']
x_d = np.arange(len(labels))
width = 0.35

y_base_d = [0.08, 0.12, 0.16, 0.16, 0.16]
y_ssd_d = [0.36, 0.51, 0.51, 0.57, 0.72]
multipliers = ['5x', '4x', '3x', '4x', '5x']

rects1 = ax.bar(x_d - width/2 - 0.02, y_base_d, width, color=C_BASE)
rects2 = ax.bar(x_d + width/2 + 0.02, y_ssd_d, width, color=C_SSD)

# Add multiplier text above SSD bars
for i, rect in enumerate(rects2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
            multipliers[i], ha='center', va='bottom', 
            color=C_SSD, fontweight='bold', fontsize=11)

ax.set_xticks(x_d)
ax.set_xticklabels(labels, color='#555555')
ax.set_ylim(0, 0.85)
ax.set_xlabel('Probability mass in top 20 tokens', fontsize=12, labelpad=10)
ax.set_ylabel('Entropy after filtering (nats)', fontsize=12)

# ==========================================
# Global Legend & Subplot Labels
# ==========================================
# Add the a, b, c, d circle labels manually
for i, ax in enumerate(axes):
    ax.tick_params(colors='#666666')
    ax.text(-0.1, -0.15, chr(97+i), transform=ax.transAxes, 
            fontsize=10, fontweight='bold', color='#666666',
            bbox=dict(facecolor='#e5e7eb', edgecolor='none', boxstyle='circle,pad=0.3'))

# Common Legend at the bottom
lines = [plt.Line2D([0], [0], color=C_BASE, lw=2),
         plt.Line2D([0], [0], color=C_SSD, lw=2)]
fig.legend(lines, ['Qwen3-30B-Instruct', 'SSD'], 
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=2, frameon=False, fontsize=14, labelcolor='#555555')

plt.savefig('figure6_entropy_distributions.png', dpi=300, bbox_inches='tight')
plt.show()