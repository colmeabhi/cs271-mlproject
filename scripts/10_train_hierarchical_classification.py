#!/usr/bin/env python3
"""
Hierarchical Fish Classification Training

Two-stage approach:
  Stage 1: Train regulatory category classifier (prohibited/restricted/legal) - 3 classes
  Stage 2: Train species classifiers for each category (12/11/4 species per category)

Output: Species name + exact regulatory restrictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter
import json

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 0.0001
WEIGHT_DECAY = 0.01

EPOCHS_REGULATORY = 20  # Stage 1: regulatory classifier
EPOCHS_SPECIES = 30     # Stage 2: species classifiers
PATIENCE = 10           # Early stopping

# ============================================================
# Data Loading
# ============================================================

def get_transforms():
    """Create data transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def calculate_class_weights(dataset, device):
    """Calculate weights for imbalanced classes"""
    class_counts = Counter([label for _, label in dataset.imgs])
    total = len(dataset)
    num_classes = len(dataset.classes)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
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


def create_model(num_classes, device):
    """Create ResNet50 model"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model.to(device)

# ============================================================
# Stage 1: Regulatory Classifier
# ============================================================

def train_regulatory_classifier(device):
    """Train 3-class regulatory category classifier"""
    print("\n" + "="*60)
    print("STAGE 1: Training Regulatory Classifier (3 classes)")
    print("="*60)
    
    train_transform, val_transform = get_transforms()
    
    # Load data
    train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Categories: {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model(num_classes=3, device=device)
    
    # Setup training
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS_REGULATORY):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS_REGULATORY} | "
              f"Train: {train_loss:.4f}, {train_acc:.2f}% | "
              f"Val: {val_loss:.4f}, {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes
            }, MODEL_DIR / 'regulatory_classifier.pth')
            print(f"  ✓ Saved best model ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(MODEL_DIR / 'regulatory_classifier.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✅ Regulatory classifier: {checkpoint['val_acc']:.2f}% val accuracy")
    return model, train_dataset.classes

# ============================================================
# Stage 2: Species Classifiers
# ============================================================

def train_species_classifier(category, device):
    """Train species classifier for a specific category"""
    print("\n" + "="*60)
    print(f"STAGE 2: Training Species Classifier - {category.upper()}")
    print("="*60)
    
    train_transform, val_transform = get_transforms()
    
    # Load data for this category only
    train_dataset = datasets.ImageFolder(DATA_DIR / 'train' / category, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / 'val' / category, transform=val_transform)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Species: {len(train_dataset.classes)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    num_species = len(train_dataset.classes)
    model = create_model(num_classes=num_species, device=device)
    
    # Setup training
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS_SPECIES):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS_SPECIES} | "
              f"Train: {train_loss:.4f}, {train_acc:.2f}% | "
              f"Val: {val_loss:.4f}, {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes,
                'category': category
            }, MODEL_DIR / f'species_classifier_{category}.pth')
            print(f"  ✓ Saved best model ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(MODEL_DIR / f'species_classifier_{category}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✅ Species classifier '{category}': {checkpoint['val_acc']:.2f}% val accuracy")
    return model, train_dataset.classes

# ============================================================
# Main Pipeline
# ============================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load species metadata
    with open(DATA_DIR.parent / 'metadata' / 'species_mapping.json') as f:
        species_metadata = json.load(f)
    
    # STAGE 1: Train regulatory classifier
    regulatory_model, regulatory_classes = train_regulatory_classifier(device)
    
    # STAGE 2: Train species classifiers
    species_class_names = {}
    for category in ['prohibited', 'restricted', 'legal']:
        _, class_names = train_species_classifier(category, device)
        species_class_names[category] = class_names
    
    # Save configuration for inference
    hierarchy_config = {
        'regulatory_model': 'models/regulatory_classifier.pth',
        'regulatory_classes': regulatory_classes,
        'species_models': {
            'prohibited': 'models/species_classifier_prohibited.pth',
            'restricted': 'models/species_classifier_restricted.pth',
            'legal': 'models/species_classifier_legal.pth'
        },
        'species_classes': species_class_names,
        'species_metadata': species_metadata
    }
    
    with open(MODEL_DIR / 'hierarchy_config.json', 'w') as f:
        json.dump(hierarchy_config, f, indent=2)
    
    print("\n" + "="*60)
    print("HIERARCHICAL TRAINING COMPLETE")
    print("="*60)
    print("\n✅ All models trained successfully!")
    print("\nModels saved:")
    print("  - models/regulatory_classifier.pth")
    print("  - models/species_classifier_prohibited.pth")
    print("  - models/species_classifier_restricted.pth")
    print("  - models/species_classifier_legal.pth")
    print("  - models/hierarchy_config.json")
    print("\nUse 11_predict_hierarchical.py for predictions!")


if __name__ == "__main__":
    main()
