#!/usr/bin/env python3
"""
Transfer Learning with ResNet50 (pretrained on ImageNet)
Two-stage training: 1) Train classifier only, 2) Fine-tune all layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
BATCH_SIZE = 32
IMG_SIZE = 224

# Two-stage training
CLASSIFIER_EPOCHS = 20  # Stage 1: train classifier only
FINETUNE_EPOCHS = 30    # Stage 2: fine-tune all layers
LR_STAGE1 = 0.001       # Higher learning rate for classifier
LR_STAGE2 = 0.0001      # Lower learning rate for fine-tuning
WEIGHT_DECAY = 0.01
PATIENCE = 10

# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load datasets with ImageNet normalization"""
    
    # Training: with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation/Test: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / 'test', transform=val_transform)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.classes, train_dataset.targets

# ============================================================
# Class Weights for Imbalanced Dataset
# ============================================================

def calculate_class_weights(targets, num_classes, device):
    """Calculate inverse frequency weights for class imbalance"""
    class_counts = Counter(targets)
    total_samples = len(targets)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights).to(device)

# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total

# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names, targets = load_data()
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(targets, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Load pretrained ResNet50
    print("\nLoading pretrained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    model = model.to(device)
    
    # ============================================================
    # STAGE 1: Train classifier only (freeze backbone)
    # ============================================================
    
    print("\n" + "="*60)
    print("STAGE 1: Training classifier only")
    print("="*60)
    
    # Freeze all layers except final classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.fc.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(CLASSIFIER_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # ============================================================
    # STAGE 2: Fine-tune all layers
    # ============================================================
    
    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning all layers")
    print("="*60)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, MODEL_DIR / 'resnet50_finetuned.pth')
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # ============================================================
    # Test Evaluation
    # ============================================================
    
    print("\nEvaluating on test set...")
    checkpoint = torch.load(MODEL_DIR / 'resnet50_finetuned.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
