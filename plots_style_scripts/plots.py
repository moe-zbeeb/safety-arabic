import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch

# ==========================================
# 0. Global Style Configuration
# ==========================================
def set_paper_style():
    """Applies the minimalist aesthetic of the paper."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.grid.axis'] = 'y'
    plt.rcParams['grid.color'] = '#f0f0f0'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['figure.facecolor'] = '#f8f9fa' # Very light grey background
    plt.rcParams['axes.facecolor'] = '#f8f9fa'

set_paper_style()

# Define the exact color palette seen in the paper's Figure 1
COLORS = {
    '4B_base_edge': '#dca99b',
    '4B_base_face': '#f6dfd8',
    '4B_ssd': '#bf6948',
    '30B_base_edge': '#a5b8ce',
    '30B_base_face': '#dce5ef',
    '30B_ssd': '#557cbc',
    'text_dark': '#333333',
    'text_gain': '#333333'
}

# ==========================================
# 1. Main Grouped Bar Chart (Like Figure 1)
# ==========================================
def draw_bar_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dummy Data matching the structure
    categories = ['Overall', 'Medium', 'Hard']
    x = np.arange(len(categories))
    width = 0.18
    spacing = 0.02
    
    data_4b_base = [34.6, 34.4, 10.5]
    data_4b_ssd = [42.4, 45.1, 16.2]
    data_30b_base = [42.4, 47.6, 18.3]
    data_30b_ssd = [55.3, 61.8, 33.6]

    # Draw bars
    # 4B Base (Hatched)
    rects1 = ax.bar(x - 1.5*width - 1.5*spacing, data_4b_base, width, 
                    facecolor=COLORS['4B_base_face'], edgecolor=COLORS['4B_base_edge'], 
                    hatch='xx', linewidth=1)
    # 4B SSD (Solid)
    rects2 = ax.bar(x - 0.5*width - 0.5*spacing, data_4b_ssd, width, 
                    color=COLORS['4B_ssd'])
    
    # 30B Base (Hatched)
    rects3 = ax.bar(x + 0.5*width + 0.5*spacing, data_30b_base, width, 
                    facecolor=COLORS['30B_base_face'], edgecolor=COLORS['30B_base_edge'], 
                    hatch='xx', linewidth=1)
    # 30B SSD (Solid)
    rects4 = ax.bar(x + 1.5*width + 1.5*spacing, data_30b_ssd, width, 
                    color=COLORS['30B_ssd'])

    # Add text labels on top of bars
    def autolabel(rects, color):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color=color, fontsize=9, fontweight='bold')

    autolabel(rects1, COLORS['4B_base_edge'])
    autolabel(rects2, COLORS['4B_ssd'])
    autolabel(rects3, COLORS['30B_base_edge'])
    autolabel(rects4, COLORS['30B_ssd'])

    # Add the "+Gain" brackets
    def add_gain_bracket(rect_base, rect_ssd, gain):
        for rb, rs, g in zip(rect_base, rect_ssd, gain):
            x1 = rb.get_x() + rb.get_width() / 2
            x2 = rs.get_x() + rs.get_width() / 2
            y_max = rs.get_height() + 5
            
            # Draw bracket line
            ax.plot([x1, x1, x2, x2], [rb.get_height() + 3, y_max, y_max, rs.get_height() + 3], 
                    color='#888888', linewidth=1)
            # Draw text
            ax.text((x1 + x2) / 2, y_max + 1, f'+{g:.1f}', ha='center', va='bottom', 
                    color=COLORS['text_gain'], fontsize=10)

    add_gain_bracket(rects1, rects2, np.array(data_4b_ssd) - np.array(data_4b_base))
    add_gain_bracket(rects3, rects4, np.array(data_30b_ssd) - np.array(data_30b_base))

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 75)
    
    # Custom Legend
    legend_elements = [
        Patch(facecolor=COLORS['4B_base_face'], edgecolor=COLORS['4B_base_edge'], hatch='xx', label='4B Base'),
        Patch(facecolor=COLORS['4B_ssd'], label='4B +SSD'),
        Patch(facecolor=COLORS['30B_base_face'], edgecolor=COLORS['30B_base_edge'], hatch='xx', label='30B Base'),
        Patch(facecolor=COLORS['30B_ssd'], label='30B +SSD')
    ]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4, frameon=False)
    
    plt.title("LiveCodeBench V6 (Pass@1)", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figure1_bar_chart.png', dpi=300)
    plt.show()

# ==========================================
# 2. Scatter Plot with Quadratic Fit (Fig 3a)
# ==========================================
def draw_scatter_fit():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Dummy data generation
    np.random.seed(42)
    x = np.random.uniform(0.5, 3.5, 200)
    
    # 3 curves mimicking the paper
    y_no_trunc = 45 - 5*(x - 1.5)**2 + np.random.normal(0, 2, 200)
    y_top5 = 48 - 3*(x - 1.8)**2 + np.random.normal(0, 2, 200)
    y_top10 = 47 - 2*(x - 1.6)**2 + np.random.normal(0, 2, 200)
    
    # Scatter points (low alpha to look like a cloud)
    ax.scatter(x, y_no_trunc, alpha=0.15, color='#888888', s=15)
    ax.scatter(x, y_top5, alpha=0.15, color=COLORS['4B_ssd'], s=15)
    ax.scatter(x, y_top10, alpha=0.15, color='#458b5e', s=15) # Greenish
    
    # Quadratic fits
    x_fit = np.linspace(0.4, 3.6, 100)
    
    for y_data, color, label in zip([y_no_trunc, y_top5, y_top10], 
                                    ['#555555', COLORS['4B_ssd'], '#458b5e'],
                                    ['no trunc.', 'top-k=5', 'top-k=10']):
        coeffs = np.polyfit(x, y_data, 2)
        p = np.poly1d(coeffs)
        ax.plot(x_fit, p(x_fit), color=color, linewidth=2, label=label)

    # Baseline dotted line
    ax.axhline(42.4, color='#888888', linestyle='--', label='baseline 42.4%')
    
    ax.set_xlabel(r'$T_{train} \times T_{eval}$', fontweight='bold')
    ax.set_ylabel('pass@1 (%)', fontweight='bold')
    ax.legend(frameon=True, edgecolor='#cccccc', facecolor='#ffffff')
    
    plt.tight_layout()
    plt.savefig('figure3a_scatter.png', dpi=300)
    plt.show()

# ==========================================
# 3. Heatmap Matrix (Like Fig 3b/9/10)
# ==========================================
def draw_heatmap():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Dummy grid data
    eval_temps = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    train_temps = [0.5, 0.7, 1.0, 1.5, 2.0]
    
    data = np.random.uniform(40, 55, size=(len(train_temps), len(eval_temps)))
    # Add a "hotspot" diagonal to mimic the effective temperature curve
    for i in range(len(train_temps)):
        for j in range(len(eval_temps)):
            if abs(train_temps[i] * eval_temps[j] - 1.2) < 0.4:
                data[i, j] += 5
    
    df = pd.DataFrame(data, index=train_temps[::-1], columns=eval_temps)
    
    # The paper uses a diverging colormap: orange (low) to white to dark blue (high)
    cmap = sns.diverging_palette(30, 250, s=80, l=55, as_cmap=True)
    
    sns.heatmap(df, annot=True, fmt=".1f", cmap=cmap, cbar=True, 
                linewidths=1, linecolor='white',
                annot_kws={"size": 9, "weight": "bold"}, ax=ax)
    
    ax.set_xlabel('Eval Temperature ($T_{eval}$)', fontweight='bold')
    ax.set_ylabel('Training Temperature ($T_{train}$)', fontweight='bold')
    
    plt.title("Best pass@1 across iterations", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figure3b_heatmap.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_bar_chart()
    draw_scatter_fit()
    draw_heatmap()