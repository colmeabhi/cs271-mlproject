"""
Model Performance Comparison
Visualizes validation accuracy across all trained models
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model performance data
models = ['Basic CNN\n(No Aug)', 'Basic CNN\n(With Aug)', 'ResNet50\n(No Aug)', 'ResNet50\n(With Aug)', 'Hierarchical\n(2-Stage)']
val_accuracy = [57.4, 54.4, 85.9, 89.9, 91.2]  # Validation accuracy
test_accuracy = [56.4, 56.0, 85.5, None, None]  # Test accuracy (where available)
train_val_gap = [3.4, 0.6, 14.1, 9.8, 7.5]  # Train/val gap (overfitting measure)

# Color scheme
colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c', '#27ae60']
edge_colors = ['#2980b9', '#2980b9', '#c0392b', '#c0392b', '#229954']

# Setup output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'evaluation_results' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================
# Plot 1: Validation Accuracy Comparison
# ============================================================
x_pos = np.arange(len(models))
bars = ax1.bar(x_pos, val_accuracy, color=colors, edgecolor=edge_colors, linewidth=2, alpha=0.85)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, val_accuracy)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add test accuracy if available
    if test_accuracy[i] is not None:
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'Test: {test_accuracy[i]:.1f}%',
                 ha='center', va='center', fontsize=9, style='italic', color='white')

ax1.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
ax1.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance Comparison\nValidation Accuracy', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
ax1.legend(loc='upper left', fontsize=10)

# Add annotation for improvement
ax1.annotate('', xy=(4, val_accuracy[4]), xytext=(0, val_accuracy[0]),
            arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.3))
ax1.text(2, 75, f'+{val_accuracy[4] - val_accuracy[0]:.1f}%\nimprovement',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# ============================================================
# Plot 2: Overfitting Analysis (Train/Val Gap)
# ============================================================
bars2 = ax2.bar(x_pos, train_val_gap, color=colors, edgecolor=edge_colors, linewidth=2, alpha=0.85)

# Add value labels
for bar, val in zip(bars2, train_val_gap):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{val:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
ax2.set_ylabel('Train/Val Gap (%)', fontsize=13, fontweight='bold')
ax2.set_title('Overfitting Analysis\nTrain-Validation Accuracy Gap (Lower is Better)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_ylim(0, 16)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='<5% Target (Good)')
ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='<10% (Acceptable)')
ax2.legend(loc='upper right', fontsize=9)

# Add annotation for best generalization
best_idx = np.argmin(train_val_gap)
ax2.annotate('Best\nGeneralization',
            xy=(best_idx, train_val_gap[best_idx]), xytext=(best_idx-0.5, train_val_gap[best_idx]+3),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=10, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

plt.suptitle('Fish Classification: Model Performance & Generalization Analysis',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / '07_model_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print summary
print(f"\n{'='*70}")
print(f"Model Performance Summary")
print(f"{'='*70}")
print(f"\n{'Model':<25} {'Val Acc':<12} {'Test Acc':<12} {'Train/Val Gap':<15}")
print(f"{'-'*70}")
for i, model in enumerate(models):
    model_clean = model.replace('\n', ' ')
    test_str = f"{test_accuracy[i]:.1f}%" if test_accuracy[i] is not None else "N/A"
    print(f"{model_clean:<25} {val_accuracy[i]:<12.1f}% {test_str:<12} {train_val_gap[i]:<15.1f}%")

print(f"\n{'='*70}")
print(f"Key Findings:")
print(f"{'='*70}")
print(f"  • Best Accuracy: {models[np.argmax(val_accuracy)].replace(chr(10), ' ')} ({max(val_accuracy):.1f}%)")
print(f"  • Best Generalization: {models[np.argmin(train_val_gap)].replace(chr(10), ' ')} ({min(train_val_gap):.1f}% gap)")
print(f"  • Improvement over Baseline: +{max(val_accuracy) - val_accuracy[0]:.1f}%")
print(f"  • Transfer Learning Impact: +{val_accuracy[2] - val_accuracy[0]:.1f}%")
print(f"  • Data Augmentation Impact: +{val_accuracy[3] - val_accuracy[2]:.1f}% (ResNet)")
print(f"  • Hierarchical Advantage: +{val_accuracy[4] - val_accuracy[3]:.1f}% over best ResNet")
print(f"{'='*70}")
