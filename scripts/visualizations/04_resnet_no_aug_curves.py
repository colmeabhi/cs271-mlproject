"""
Training Curves Visualization for ResNet50 Transfer Learning (No Augmentation)

Plots accuracy and loss curves for both training stages:
- Stage 1: Classifier only (20 epochs)
- Stage 2: Fine-tuning all layers (19 epochs, early stopping)

Output: Saves figure to evaluation_results/visualizations/
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Stage 1: Training classifier only (20 epochs)
stage1_epochs = list(range(1, 21))

stage1_train_loss = [1.0425, 0.9170, 0.8747, 0.8544, 0.8344, 0.8298, 0.8484, 0.8402, 0.8415, 0.8376,
                     0.8458, 0.8469, 0.8423, 0.8441, 0.8192, 0.8459, 0.8194, 0.8366, 0.8251, 0.8288]

stage1_train_acc = [45.72, 55.33, 58.83, 59.30, 61.71, 61.21, 59.39, 61.45, 61.27, 60.92,
                    60.92, 61.92, 61.71, 62.24, 63.71, 61.71, 62.44, 60.95, 63.09, 61.97]

stage1_val_loss = [1.0079, 0.8207, 0.8261, 0.8533, 0.8488, 0.8081, 0.8366, 0.8015, 0.7965, 0.7869,
                   0.8420, 0.7395, 0.8547, 0.9670, 0.8538, 0.7319, 0.7510, 0.7331, 0.8613, 0.7639]

stage1_val_acc = [48.32, 64.03, 64.43, 62.42, 61.21, 63.89, 63.22, 66.17, 65.37, 65.91,
                  62.82, 67.38, 61.48, 57.85, 63.36, 68.46, 66.71, 67.65, 62.28, 66.44]

# Stage 2: Fine-tuning all layers (19 epochs, stopped early)
stage2_epochs = list(range(1, 20))

stage2_train_loss = [0.6892, 0.2380, 0.1271, 0.1257, 0.1144, 0.1079, 0.0660, 0.0270, 0.0103, 0.0099,
                     0.0085, 0.0042, 0.0047, 0.0023, 0.0030, 0.0020, 0.0039, 0.0028, 0.0014]

stage2_train_acc = [70.00, 90.57, 95.30, 94.83, 95.50, 95.86, 97.71, 99.03, 99.68, 99.76,
                    99.79, 99.88, 99.94, 99.94, 99.91, 99.94, 99.91, 99.94, 99.97]

stage2_val_loss = [0.5665, 0.5599, 0.4748, 0.7829, 0.7712, 0.5687, 0.6488, 0.5563, 0.5640, 0.5503,
                   0.6271, 0.6115, 0.6063, 0.6266, 0.6100, 0.5825, 0.5764, 0.5688, 0.5781]

stage2_val_acc = [76.64, 81.34, 83.22, 77.58, 81.34, 82.95, 83.62, 85.37, 85.91, 85.50,
                  85.23, 84.97, 85.77, 84.56, 85.77, 84.97, 85.37, 85.91, 85.64]

test_acc = 85.47

# Setup output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'evaluation_results' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create figure with 2x2 subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# Stage 1: Accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(stage1_epochs, stage1_train_acc, 'o-', label='Train Accuracy', color='#3498db', linewidth=2, markersize=4)
ax1.plot(stage1_epochs, stage1_val_acc, 's-', label='Val Accuracy', color='#e74c3c', linewidth=2, markersize=4)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Stage 1: Classifier Only\nAccuracy per Epoch', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 21)

# Stage 1: Loss
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(stage1_epochs, stage1_train_loss, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=4)
ax2.plot(stage1_epochs, stage1_val_loss, 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Stage 1: Classifier Only\nLoss per Epoch', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 21)

# Stage 2: Accuracy
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(stage2_epochs, stage2_train_acc, 'o-', label='Train Accuracy', color='#3498db', linewidth=2, markersize=4)
ax3.plot(stage2_epochs, stage2_val_acc, 's-', label='Val Accuracy', color='#e74c3c', linewidth=2, markersize=4)
ax3.axhline(y=test_acc, color='#27ae60', linestyle='--', linewidth=2, label=f'Test Accuracy ({test_acc:.2f}%)')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Stage 2: Fine-tuning All Layers\nAccuracy per Epoch (Early Stopped)', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 20)

# Stage 2: Loss
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(stage2_epochs, stage2_train_loss, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=4)
ax4.plot(stage2_epochs, stage2_val_loss, 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=4)
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax4.set_title('Stage 2: Fine-tuning All Layers\nLoss per Epoch (Early Stopped)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 20)

# Add overall title
fig.suptitle('ResNet50 Transfer Learning - No Augmentation\nTwo-Stage Training Process',
             fontsize=15, fontweight='bold', y=0.995)

# Save
output_path = OUTPUT_DIR / '04_resnet_no_aug_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print summary
print(f"\nTraining Summary (ResNet50 Transfer Learning - No Augmentation):")
print(f"\nStage 1 (Classifier Only - 20 epochs):")
print(f"  Final Train Accuracy: {stage1_train_acc[-1]:.2f}%")
print(f"  Final Val Accuracy: {stage1_val_acc[-1]:.2f}%")
print(f"  Best Val Accuracy: {max(stage1_val_acc):.2f}% (Epoch {stage1_val_acc.index(max(stage1_val_acc)) + 1})")

print(f"\nStage 2 (Fine-tuning All Layers - 19 epochs, early stopped):")
print(f"  Final Train Accuracy: {stage2_train_acc[-1]:.2f}%")
print(f"  Final Val Accuracy: {stage2_val_acc[-1]:.2f}%")
print(f"  Best Val Accuracy: {max(stage2_val_acc):.2f}% (Epoch {stage2_val_acc.index(max(stage2_val_acc)) + 1})")
print(f"  Test Accuracy: {test_acc:.2f}%")
print(f"\n  Train/Val Gap at Best: {stage2_train_acc[stage2_val_acc.index(max(stage2_val_acc))]:.2f}% - {max(stage2_val_acc):.2f}% = {stage2_train_acc[stage2_val_acc.index(max(stage2_val_acc))] - max(stage2_val_acc):.2f}%")
