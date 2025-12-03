"""
Training Curves Visualization for Basic CNN (With Augmentation)

Plots accuracy and loss curves across epochs for 07_train.py
Output: Saves figure to evaluation_results/visualizations/
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Training data from 07_train.py
epochs = list(range(1, 26))

train_loss = [2.3272, 1.0601, 1.0516, 1.0337, 1.0342, 1.0305, 1.0258, 1.0352, 1.0160, 1.0198,
              1.0119, 1.0142, 1.0123, 0.9920, 0.9880, 0.9805, 0.9767, 0.9615, 0.9551, 0.9529,
              0.9519, 0.9630, 0.9399, 0.9359, 0.9324]

train_acc = [43.37, 45.99, 47.72, 48.49, 49.87, 49.78, 48.57, 49.78, 49.81, 49.40,
             49.43, 49.63, 50.07, 50.60, 51.60, 50.93, 52.39, 51.48, 53.81, 53.13,
             52.92, 53.04, 54.07, 54.89, 55.22]

val_loss = [1.0480, 1.0375, 1.0122, 1.0168, 1.0105, 1.0205, 1.0015, 1.0042, 0.9992, 1.0031,
            1.0049, 0.9998, 1.0033, 0.9809, 0.9782, 0.9642, 0.9675, 0.9638, 0.9557, 1.0002,
            0.9811, 0.9605, 0.9562, 0.9587, 0.9441]

val_acc = [50.74, 50.34, 50.20, 47.65, 51.28, 46.98, 52.35, 50.74, 51.68, 50.60,
           50.34, 51.68, 51.14, 53.02, 51.81, 53.29, 53.56, 52.89, 52.35, 51.68,
           53.02, 53.83, 55.84, 54.63, 54.50]

test_acc = 56.01

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
ax1.set_title('Basic CNN - With Augmentation\nAccuracy per Epoch', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 26)

# Plot 2: Loss
ax2.plot(epochs, train_loss, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=4)
ax2.plot(epochs, val_loss, 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Basic CNN - With Augmentation\nLoss per Epoch', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 26)

plt.tight_layout()

# Save
output_path = OUTPUT_DIR / '03_train_aug_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print summary
print(f"\nTraining Summary (Basic CNN - With Augmentation):")
print(f"  Final Train Accuracy: {train_acc[-1]:.2f}%")
print(f"  Final Val Accuracy: {val_acc[-1]:.2f}%")
print(f"  Best Val Accuracy: {max(val_acc):.2f}% (Epoch {val_acc.index(max(val_acc)) + 1})")
print(f"  Test Accuracy: {test_acc:.2f}%")
print(f"  Final Train Loss: {train_loss[-1]:.4f}")
print(f"  Final Val Loss: {val_loss[-1]:.4f}")
