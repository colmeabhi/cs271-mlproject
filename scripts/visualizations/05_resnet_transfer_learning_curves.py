"""
Training Curves Visualization for ResNet50 Transfer Learning (With Augmentation)

Plots accuracy and loss curves for both training stages:
- Stage 1: Classifier only (20 epochs)
- Stage 2: Fine-tuning all layers (25 epochs, continued training)

Output: Saves figure to evaluation_results/visualizations/
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Stage 1: Training classifier only (20 epochs)
stage1_epochs = list(range(1, 21))

stage1_train_loss = [1.0775, 0.9527, 0.9230, 0.9044, 0.8991, 0.8836, 0.8811, 0.9235, 0.9193, 0.8884,
                     0.8965, 0.8787, 0.8982, 0.8655, 0.8667, 0.8662, 0.8571, 0.8464, 0.8405, 0.8472]

stage1_train_acc = [43.61, 53.78, 55.39, 57.57, 57.10, 58.83, 57.95, 56.27, 56.30, 57.63,
                    57.77, 58.45, 58.10, 59.39, 58.83, 60.06, 59.45, 60.09, 60.86, 60.71]

stage1_val_loss = [1.1520, 1.0769, 1.0165, 0.8825, 0.8570, 0.8444, 0.9015, 0.9528, 0.7950, 0.8990,
                   0.8852, 0.8001, 0.8390, 0.8039, 0.8636, 0.8800, 0.8885, 0.8547, 0.8282, 0.8871]

stage1_val_acc = [42.55, 46.44, 50.07, 57.05, 60.00, 59.60, 57.99, 55.03, 63.62, 56.78,
                  58.39, 63.76, 61.07, 63.22, 59.73, 57.85, 58.93, 60.27, 61.34, 58.52]

# Stage 2: Fine-tuning all layers (25 epochs shown here, but training continued)
stage2_epochs = list(range(1, 26))

stage2_train_loss = [0.7175, 0.4109, 0.3251, 0.2467, 0.2379, 0.1633, 0.1468, 0.1526, 0.0839, 0.0518,
                     0.0329, 0.0324, 0.0354, 0.0285, 0.0173, 0.0133, 0.0085, 0.0085, 0.0096, 0.0085,
                     0.0056, 0.0081, 0.0119, 0.0052, 0.0091]

stage2_train_acc = [68.67, 82.52, 87.01, 89.60, 90.86, 93.89, 94.27, 94.36, 96.77, 98.06,
                    98.82, 98.85, 98.85, 98.82, 99.35, 99.68, 99.88, 99.76, 99.68, 99.74,
                    99.94, 99.82, 99.65, 99.88, 99.68]

stage2_val_loss = [0.6075, 0.4995, 0.5460, 0.4318, 0.5158, 0.4961, 0.5804, 0.5814, 0.4415, 0.3790,
                   0.4313, 0.4453, 0.4753, 0.4865, 0.4654, 0.4708, 0.4680, 0.4712, 0.4853, 0.4423,
                   0.4552, 0.4416, 0.4446, 0.4737, 0.4382]

stage2_val_acc = [75.57, 80.00, 79.46, 83.62, 82.82, 84.97, 80.81, 83.09, 86.58, 87.65,
                  88.05, 86.04, 85.77, 86.58, 88.59, 87.92, 88.46, 88.59, 89.80, 89.93,
                  89.26, 89.93, 89.40, 89.53, 89.80]

# Best model was saved at epoch 20 with 89.93% val accuracy
best_epoch = 20
best_val_acc = 89.93

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
ax1.set_ylim(40, 70)

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
# Mark best model
ax3.plot(best_epoch, best_val_acc, '*', color='#27ae60', markersize=15, label=f'Best Model (Epoch {best_epoch}: {best_val_acc:.2f}%)')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Stage 2: Fine-tuning All Layers\nAccuracy per Epoch', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 26)
ax3.set_ylim(70, 100)

# Stage 2: Loss
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(stage2_epochs, stage2_train_loss, 'o-', label='Train Loss', color='#3498db', linewidth=2, markersize=4)
ax4.plot(stage2_epochs, stage2_val_loss, 's-', label='Val Loss', color='#e74c3c', linewidth=2, markersize=4)
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax4.set_title('Stage 2: Fine-tuning All Layers\nLoss per Epoch', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 26)

# Add overall title
fig.suptitle('ResNet50 Transfer Learning - With Data Augmentation\nTwo-Stage Training Process',
             fontsize=15, fontweight='bold', y=0.995)

# Save
output_path = OUTPUT_DIR / '05_resnet_transfer_learning_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Print summary
print(f"\nTraining Summary (ResNet50 Transfer Learning - With Augmentation):")
print(f"\nStage 1 (Classifier Only - 20 epochs):")
print(f"  Final Train Accuracy: {stage1_train_acc[-1]:.2f}%")
print(f"  Final Val Accuracy: {stage1_val_acc[-1]:.2f}%")
print(f"  Best Val Accuracy: {max(stage1_val_acc):.2f}% (Epoch {stage1_val_acc.index(max(stage1_val_acc)) + 1})")

print(f"\nStage 2 (Fine-tuning All Layers - 25 epochs shown):")
print(f"  Final Train Accuracy: {stage2_train_acc[-1]:.2f}%")
print(f"  Final Val Accuracy: {stage2_val_acc[-1]:.2f}%")
print(f"  Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"  Train/Val Gap at Best: {stage2_train_acc[best_epoch-1]:.2f}% - {best_val_acc:.2f}% = {stage2_train_acc[best_epoch-1] - best_val_acc:.2f}%")

print(f"\n📊 Key Observations:")
print(f"  • Stage 1 improved val accuracy from 42.55% → 63.76% (best)")
print(f"  • Stage 2 fine-tuning boosted performance to 89.93% (best)")
print(f"  • Model achieved 99.74% train accuracy (epoch 20)")
print(f"  • Significant overfitting visible: ~10% train/val gap")
