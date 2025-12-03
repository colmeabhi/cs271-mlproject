"""
Dataset Distribution Visualization Script

Generates class imbalance visualization showing total images per species.
Output: Saves figure to evaluation_results/visualizations/
"""

import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / 'config.yaml'
CLASS_DIST_PATH = BASE_DIR / 'data' / 'metadata' / 'class_distribution.json'
OUTPUT_DIR = BASE_DIR / 'evaluation_results' / 'visualizations'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

with open(CLASS_DIST_PATH, 'r') as f:
    class_distribution = json.load(f)

# Build species mapping
species_mapping = {}
category_mapping = {}

for category in ['prohibited', 'restricted', 'legal']:
    for species in config['species'][category]:
        code = species['code']
        species_mapping[code] = species['common']
        category_mapping[code] = category

# Calculate total images per species
species_data = []
for code in class_distribution['train'].keys():
    total = (class_distribution['train'][code] +
             class_distribution['val'][code] +
             class_distribution['test'][code])
    species_data.append({
        'code': code,
        'name': species_mapping[code],
        'category': category_mapping[code],
        'total': total
    })

# Sort by total count descending
species_data.sort(key=lambda x: x['total'], reverse=True)

# Extract data for plotting
species_names = [s['name'] for s in species_data]
totals = [s['total'] for s in species_data]

# Color by category
category_colors = {
    'prohibited': '#e74c3c',
    'restricted': '#f39c12',
    'legal': '#27ae60'
}
colors = [category_colors[s['category']] for s in species_data]

# Create plot
fig, ax = plt.subplots(figsize=(12, 10))

bars = ax.barh(species_names, totals, color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Total Number of Images', fontsize=12, fontweight='bold')
ax.set_ylabel('Species', fontsize=12, fontweight='bold')
ax.set_title('Dataset Distribution: Total Images per Species', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, count) in enumerate(zip(bars, totals)):
    ax.text(count + 3, i, str(count), va='center', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=category_colors['prohibited'], edgecolor='black', label='Prohibited'),
    Patch(facecolor=category_colors['restricted'], edgecolor='black', label='Restricted'),
    Patch(facecolor=category_colors['legal'], edgecolor='black', label='Legal')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add statistics text
stats_text = (f"Max: {max(totals)} | Min: {min(totals)} | "
             f"Mean: {np.mean(totals):.1f} | Median: {np.median(totals):.1f} | "
             f"Imbalance Ratio: {max(totals)/min(totals):.2f}x")
ax.text(0.5, -0.05, stats_text, transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / '01_class_imbalance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print summary
print(f"\nDataset Summary:")
print(f"  Total species: {len(species_data)}")
print(f"  Total images: {sum(totals)}")
print(f"  Max: {max(totals)} | Min: {min(totals)} | Mean: {np.mean(totals):.1f}")
print(f"  Imbalance ratio: {max(totals)/min(totals):.2f}x")
