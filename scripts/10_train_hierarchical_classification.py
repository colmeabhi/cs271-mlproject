#!/usr/bin/env python3
"""
10_train_hierarchical_classification.py - Hierarchical Fish Classification

Two-stage prediction pipeline:
  Stage 1: Predict regulatory category (prohibited/restricted/legal) - 3 classes
  Stage 2: Predict specific species within category - 12/12/4 classes per category

Final output: Species name + exact regulatory restrictions

This approach combines:
- High accuracy regulatory classification (simpler 3-class problem)
- Detailed species identification (easier sub-problems: 12, 12, or 4 classes)
- Practical output with specific fishing regulations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from pathlib import Path
from collections import Counter
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS_REGULATORY = 20   # Train regulatory classifier
EPOCHS_SPECIES = 30      # Train species classifiers
LR = 0.0001
WEIGHT_DECAY = 0.01
IMG_SIZE = 224
PATIENCE = 10


def get_data_transforms():
    """Create data transforms with augmentation"""
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


def create_resnet50_model(num_classes, device):
    """Create ResNet50 model with custom classifier"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )

    return model.to(device)


def train_regulatory_classifier(device):
    """
    STAGE 1: Train regulatory category classifier (3 classes)
    Returns: trained model
    """
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: Training Regulatory Classifier (3 classes)")
    logger.info("="*70)

    train_transform, val_transform = get_data_transforms()

    # Load datasets organized by category
    train_dataset = datasets.ImageFolder('data/processed/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/processed/val', transform=val_transform)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    logger.info(f"Categories: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model = create_resnet50_model(num_classes=3, device=device)

    # Loss and optimizer
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS_REGULATORY):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS_REGULATORY}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes
            }, 'models/regulatory_classifier.pth')
            logger.info(f"  ✓ Saved (best val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  Early stopping after {epoch+1} epochs")
                break

    # Load best model
    checkpoint = torch.load('models/regulatory_classifier.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"\n✅ Regulatory classifier trained: {checkpoint['val_acc']:.2f}% val accuracy")

    return model, train_dataset.classes


def train_species_classifier(category, device):
    """
    STAGE 2: Train species classifier for a specific category

    Args:
        category: 'prohibited', 'restricted', or 'legal'

    Returns: trained model, class names
    """
    logger.info("\n" + "="*70)
    logger.info(f"STAGE 2: Training Species Classifier for '{category.upper()}'")
    logger.info("="*70)

    train_transform, val_transform = get_data_transforms()

    # Load only images from this category
    train_dir = f'data/processed/train/{category}'
    val_dir = f'data/processed/val/{category}'

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    logger.info(f"Species: {len(train_dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    num_species = len(train_dataset.classes)
    model = create_resnet50_model(num_classes=num_species, device=device)

    # Loss and optimizer
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS_SPECIES):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS_SPECIES}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes,
                'category': category
            }, f'models/species_classifier_{category}.pth')
            logger.info(f"  ✓ Saved (best val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  Early stopping after {epoch+1} epochs")
                break

    # Load best model
    checkpoint = torch.load(f'models/species_classifier_{category}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"\n✅ Species classifier for '{category}' trained: {checkpoint['val_acc']:.2f}% val accuracy")

    return model, train_dataset.classes


def main():
    """Main training pipeline"""

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

    # Load species metadata
    with open('data/metadata/species_mapping.json') as f:
        species_metadata = json.load(f)

    # STAGE 1: Train regulatory classifier
    regulatory_model, regulatory_classes = train_regulatory_classifier(device)

    # STAGE 2: Train species classifiers for each category
    species_models = {}
    species_class_names = {}

    for category in ['prohibited', 'restricted', 'legal']:
        model, class_names = train_species_classifier(category, device)
        species_models[category] = model
        species_class_names[category] = class_names

    # Save all model paths and metadata for inference
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

    with open('models/hierarchy_config.json', 'w') as f:
        json.dump(hierarchy_config, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("HIERARCHICAL TRAINING COMPLETE")
    logger.info("="*70)
    logger.info("\n✅ All models trained successfully!")
    logger.info("\nModels saved:")
    logger.info("  - models/regulatory_classifier.pth (3-class regulatory)")
    logger.info("  - models/species_classifier_prohibited.pth (12 species)")
    logger.info("  - models/species_classifier_restricted.pth (12 species)")
    logger.info("  - models/species_classifier_legal.pth (4 species)")
    logger.info("  - models/hierarchy_config.json (inference configuration)")
    logger.info("\nUse 11_predict_hierarchical.py for two-stage predictions!")


if __name__ == "__main__":
    main()
