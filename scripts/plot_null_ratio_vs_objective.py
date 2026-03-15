#!/usr/bin/env python3
"""Plot: Null Ratio (x) vs Incremental Advantage (y = -objective).
Only best combo per dataset, only incremental wins (negative objective)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# --- Best combo per dataset from OVERALL SUMMARY (only incremental wins) ---
# Format: (dataset_name, combo_name, objective, N_no_ext, N_ext)
data = [
    ("BankLoanSta",       "CS+AI ext",                -0.006532, 3831,  16169),
    ("Weather",           "Evapor+Sunshi+WindDi ext",  -0.009676, 578,   4422),
    ("DiabetesRecord",    "weight+medica+A1Cres ext",  -0.005322, 8117,  12237),
    ("HRAnalytics",       "gender ext",                -0.000177, 902,   2930),
    ("ClientRecord (Aug)","Offer ext",                 -0.000341, 775,   634),
    ("Movie (Aug v2)",    "rating+rating ext",         -0.004046, 5616,  16878),
]

colors = {
    'BankLoanSta':        '#e74c3c',
    'Weather':            '#3498db',
    'DiabetesRecord':     '#2ecc71',
    'HRAnalytics':        '#9b59b6',
    'ClientRecord (Aug)': '#f39c12',
    'Movie (Aug v2)':     '#1abc9c',
}

markers = {
    'BankLoanSta':        'o',
    'Weather':            's',
    'DiabetesRecord':     '^',
    'HRAnalytics':        'D',
    'ClientRecord (Aug)': 'v',
    'Movie (Aug v2)':     'P',
}

fig, ax = plt.subplots(figsize=(12, 7))

for ds, combo, obj, n_no, n_ext in data:
    null_ratio = n_no / (n_no + n_ext) * 100
    flipped_obj = -obj  # positive = incremental advantage

    ax.scatter(null_ratio, flipped_obj,
               c=colors[ds], marker=markers[ds],
               s=180, alpha=0.9, edgecolors='black', linewidth=1.0,
               zorder=5, label=ds)

    # Label each point with dataset name
    offset_x, offset_y = 8, 0
    ha = 'left'
    # Adjust label positions to avoid overlap
    if ds == 'HRAnalytics':
        offset_y = -0.0004
    elif ds == 'ClientRecord (Aug)':
        offset_y = 0.0004
    elif ds == 'Movie (Aug v2)':
        offset_x = -8
        ha = 'right'

    ax.annotate(f'{ds}\n({combo}, obj={obj:.4f})',
                xy=(null_ratio, flipped_obj),
                xytext=(offset_x, offset_y),
                textcoords='offset points' if offset_y == 0 else 'data',
                fontsize=8.5, color=colors[ds],
                fontweight='bold',
                ha=ha, va='center')

# Horizontal line at y=0
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Null Ratio (% rows without extended features)', fontsize=13, fontweight='bold')
ax.set_ylabel('Incremental Advantage (-objective)', fontsize=13, fontweight='bold')
ax.set_title('Best Incremental Learning Wins per Dataset\nPositive = Incremental outperforms Combined',
             fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 70)

# Make sure all points are visible with some padding
y_vals = [-obj for _, _, obj, _, _ in data]
y_min, y_max = min(y_vals), max(y_vals)
y_pad = (y_max - y_min) * 0.2
ax.set_ylim(y_min - y_pad, y_max + y_pad)

plt.tight_layout()
plt.savefig('results/null_ratio_vs_objective.png', dpi=150, bbox_inches='tight')
print("Saved: results/null_ratio_vs_objective.png")
plt.close()
