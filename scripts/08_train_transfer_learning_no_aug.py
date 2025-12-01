#!/usr/bin/env python3
"""
08_train_transfer_learning_no_aug.py - Transfer Learning with ResNet50 (NO AUGMENTATION)

Uses pretrained ResNet50 WITHOUT data augmentation for baseline comparison.
Two-stage training:
  Stage 1: Train only the classifier (backbone frozen) - 10 epochs
  Stage 2: Fine-tune entire model (all layers unfrozen) - 30 epochs

Purpose: Compare impact of data augmentation on transfer learning performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 32
FREEZE_EPOCHS = 10      # Stage 1: train classifier only
FINETUNE_EPOCHS = 30    # Stage 2: fine-tune all layers
LR_STAGE1 = 0.001       # Higher LR for classifier training
LR_STAGE2 = 0.0001      # Lower LR for fine-tuning
WEIGHT_DECAY = 0.01
IMG_SIZE = 224
PATIENCE = 10           # Early stopping


def get_data_transforms():
    """Create data transforms WITHOUT augmentation (resize + normalize only)"""
    # No augmentation - same transform for train and val
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    return transform, transform  # Same for train and val


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter()
    for _, label in dataset.imgs:
        class_counts[label] += 1

    total_samples = len(dataset)
    num_classes = len(dataset.classes)

    weights = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        count = class_counts[class_idx]
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)

    logger.info(f"Class weights: min={weights.min():.2f}, max={weights.max():.2f}")
    return weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def main():
    """Main training function"""

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Load datasets (NO AUGMENTATION)
    logger.info("Loading datasets (NO augmentation)...")
    train_transform, val_transform = get_data_transforms()

    train_dataset = datasets.ImageFolder('data/processed/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/processed/val', transform=val_transform)
    test_dataset = datasets.ImageFolder('data/processed/test', transform=val_transform)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    logger.info(f"Classes: {len(train_dataset.classes)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create ResNet50 model with pretrained weights
    logger.info("Loading pretrained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace final layer for our 28 classes
    num_classes = len(train_dataset.classes)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    model = model.to(device)

    # Loss function with class weights
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ============ STAGE 1: Train classifier only ============
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: Training classifier (backbone frozen, NO augmentation)")
    logger.info("="*70)

    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.fc.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0

    for epoch in range(FREEZE_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{FREEZE_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes
            }, 'models/resnet50_no_aug_stage1.pth')
            logger.info(f"  ✓ Saved (best val_acc: {val_acc:.2f}%)")

    # ============ STAGE 2: Fine-tune entire model ============
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: Fine-tuning entire model (all layers unfrozen, NO augmentation)")
    logger.info("="*70)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(FINETUNE_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes
            }, 'models/resnet50_transfer_no_aug_best.pth')
            logger.info(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  Early stopping after {epoch+1} epochs")
                break

    # ============ FINAL TEST EVALUATION ============
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*70)

    checkpoint = torch.load('models/resnet50_transfer_no_aug_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    logger.info(f"   Final test accuracy: {test_acc:.2f}%")
    logger.info(f"   Model saved: models/resnet50_transfer_no_aug_best.pth")
    logger.info(f"\nℹ️  Compare with augmentation version to measure impact!")


if __name__ == "__main__":
    main()
