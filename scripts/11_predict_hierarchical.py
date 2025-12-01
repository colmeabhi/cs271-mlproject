#!/usr/bin/env python3
"""
Two-Stage Hierarchical Fish Prediction

Pipeline:
  1. Predict regulatory category (prohibited/restricted/legal)
  2. Load appropriate species classifier for that category
  3. Predict specific fish species
  4. Return species name + exact regulatory restrictions

Usage:
    python scripts/11_predict_hierarchical.py <image_path>
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path('models/hierarchy_config.json')
IMG_SIZE = 224

# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path, num_classes, device):
    """Load a trained ResNet50 model"""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

# ============================================================
# Hierarchical Classifier
# ============================================================

class HierarchicalFishClassifier:
    """Two-stage hierarchical fish classifier"""
    
    def __init__(self, config_path=CONFIG_PATH, device=None):
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else
                                'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Load regulatory classifier (3 classes)
        self.regulatory_model = load_model(
            self.config['regulatory_model'],
            num_classes=3,
            device=self.device
        )
        self.regulatory_classes = self.config['regulatory_classes']
        
        # Load species classifiers for each category
        self.species_models = {}
        for category in ['prohibited', 'restricted', 'legal']:
            num_species = len(self.config['species_classes'][category])
            self.species_models[category] = load_model(
                self.config['species_models'][category],
                num_classes=num_species,
                device=self.device
            )
        
        self.species_classes = self.config['species_classes']
        self.species_metadata = self.config['species_metadata']
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ Hierarchical classifier loaded")
    
    def predict(self, image_path):
        """Two-stage prediction: category → species"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # STAGE 1: Predict regulatory category
        with torch.no_grad():
            regulatory_output = self.regulatory_model(image_tensor)
            regulatory_probs = torch.softmax(regulatory_output, dim=1)
            category_confidence, category_idx = regulatory_probs.max(1)
        
        category = self.regulatory_classes[category_idx.item()]
        category_conf = category_confidence.item() * 100
        
        # STAGE 2: Predict species within category
        species_model = self.species_models[category]
        
        with torch.no_grad():
            species_output = species_model(image_tensor)
            species_probs = torch.softmax(species_output, dim=1)
            species_confidence, species_idx = species_probs.max(1)
        
        species_code = self.species_classes[category][species_idx.item()]
        species_conf = species_confidence.item() * 100
        
        # Get species metadata
        metadata = self.species_metadata.get(species_code, {})
        
        return {
            'category': category,
            'category_confidence': category_conf,
            'species_code': species_code,
            'species_confidence': species_conf,
            'scientific_name': metadata.get('scientific_name', 'Unknown'),
            'common_name': metadata.get('common_name', 'Unknown'),
            'regulation': metadata.get('regulation', 'See current regulations')
        }
    
    def predict_and_display(self, image_path):
        """Predict and display formatted results"""
        result = self.predict(image_path)
        
        print("\n" + "="*70)
        print("FISH IDENTIFICATION RESULTS")
        print("="*70)
        print(f"\nImage: {image_path}")
        print(f"\nRegulatory Category: {result['category'].upper()}")
        print(f"  Confidence: {result['category_confidence']:.1f}%")
        print(f"\nSpecies: {result['common_name']}")
        print(f"  Scientific Name: {result['scientific_name']}")
        print(f"  Species Code: {result['species_code']}")
        print(f"  Confidence: {result['species_confidence']:.1f}%")
        print(f"\nREGULATION:")
        
        # Color-coded regulation output
        if result['category'] == 'prohibited':
            print(f"  🚫 {result['regulation']}")
        elif result['category'] == 'restricted':
            print(f"  ⚠️  {result['regulation']}")
        else:
            print(f"  ✅ {result['regulation']}")
        
        print("="*70 + "\n")
        
        return result

# ============================================================
# Command-Line Interface
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/11_predict_hierarchical.py <image_path>")
        print("\nExample:")
        print("  python scripts/11_predict_hierarchical.py data/processed/test/prohibited/yelloweye/img_001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Initialize classifier and predict
    classifier = HierarchicalFishClassifier()
    classifier.predict_and_display(image_path)


if __name__ == "__main__":
    main()
