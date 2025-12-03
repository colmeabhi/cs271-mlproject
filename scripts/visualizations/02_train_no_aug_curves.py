"""
Training Curves Visualization for Basic CNN (No Augmentation)

Plots accuracy and loss curves across epochs for 07_train_no_aug.py
Output: Saves figure to evaluation_results/visualizations/
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Training data from 07_train_no_aug.py
epochs = list(range(1, 26))

train_loss = [2.6478, 1.0164, 0.9963, 0.9840, 0.9813, 0.9786, 0.9512, 0.9591, 0.9501, 0.9636,
              0.9456, 0.9215, 0.9184, 0.9219, 0.9330, 0.8997, 0.9071, 0.9009, 0.9026, 0.8794,
              0.8821, 0.8517, 0.8283, 0.8095, 0.8132]

train_acc = [41.76, 48.34, 49.84, 50.48, 51.28, 52.01, 52.81, 52.19, 53.78, 54.07,
             53.01, 54.89, 54.28, 54.51, 53.45, 54.86, 55.25, 56.42, 56.07, 56.83,
             57.80, 58.83, 60.42, 60.59, 60.42]

val_loss = [1.0228, 1.0195, 1.0119, 0.9874, 0.9754, 0.9823, 0.9627, 0.9667, 0.9728, 0.9646,
            0.9500, 0.9588, 0.9604, 0.9340, 0.9388, 0.9273, 0.9215, 0.9255, 0.9363, 0.9392,
            0.9669, 0.9074, 0.8978, 0.8933, 0.8862]

val_acc = [46.58, 46.98, 48.72, 50.34, 51.81, 53.02, 54.36, 52.35, 52.62, 53.15,
           53.02, 54.77, 54.63, 54.09, 54.36, 55.30, 54.63, 55.84, 54.23, 56.64,
           54.09, 56.78, 57.18, 56.78, 56.51]

test_acc = 56.41

# Setup output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'evaluation_results' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy
ax1.plot(epochs, train_acc, 'o-', label='Train Accuracy', color='#3498db', linewidth=2, markersize=4)
ax1.plot(epochs, val_acc, 's-', label='Val Accuracy', color='#e74c3c', linewidth=2, markersize=4)
ax1.axhline(y=test_acc, color='#27ae60', linestyle='--', linewidth=2, label=f'Test Accuracy ({test_acc:.2f}%)')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Basic CNN - No Augmentation\nAccuracy per Epoch', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 26)

# Plot 2: Loss
ax2.plot(epochs, train_loss, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=4)
ax2.plot(epochs, val_loss, 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Basic CNN - No Augmentation\nLoss per Epoch', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 26)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / '02_train_no_aug_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print summary
print(f"\nTraining Summary (Basic CNN - No Augmentation):")
print(f"  Final Train Accuracy: {train_acc[-1]:.2f}%")
print(f"  Final Val Accuracy: {val_acc[-1]:.2f}%")
print(f"  Best Val Accuracy: {max(val_acc):.2f}% (Epoch {val_acc.index(max(val_acc)) + 1})")
print(f"  Test Accuracy: {test_acc:.2f}%")
print(f"  Final Train Loss: {train_loss[-1]:.4f}")
print(f"  Final Val Loss: {val_loss[-1]:.4f}")
