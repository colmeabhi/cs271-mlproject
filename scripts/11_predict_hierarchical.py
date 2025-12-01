#!/usr/bin/env python3
"""
11_predict_hierarchical.py - Two-Stage Hierarchical Fish Prediction

Prediction pipeline:
  1. Load image
  2. Predict regulatory category (prohibited/restricted/legal)
  3. Load appropriate species classifier for that category
  4. Predict specific fish species
  5. Return species name + exact regulatory restrictions

Usage:
    python scripts/11_predict_hierarchical.py <image_path>
    python scripts/11_predict_hierarchical.py data/processed/test/prohibited/yelloweye/img_001.jpg

Example output:
    Category: PROHIBITED
    Species: Yelloweye Rockfish (Sebastes ruberrimus)
    Regulation: Prohibited - Zero Take (Overfished)
    Confidence: Category 98.5%, Species 96.2%
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class HierarchicalFishClassifier:
    """Two-stage hierarchical fish classifier"""

    def __init__(self, config_path='models/hierarchy_config.json', device=None):
        """
        Initialize hierarchical classifier

        Args:
            config_path: Path to hierarchy configuration JSON
            device: torch device (auto-detected if None)
        """
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)

        # Load regulatory classifier
        self.regulatory_model = self._load_model(
            self.config['regulatory_model'],
            num_classes=3
        )
        self.regulatory_classes = self.config['regulatory_classes']

        # Load species classifiers
        self.species_models = {}
        for category in ['prohibited', 'restricted', 'legal']:
            model_path = self.config['species_models'][category]
            num_species = len(self.config['species_classes'][category])
            self.species_models[category] = self._load_model(model_path, num_classes=num_species)

        self.species_classes = self.config['species_classes']
        self.species_metadata = self.config['species_metadata']

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        logger.info("✅ Hierarchical classifier loaded successfully")

    def _load_model(self, checkpoint_path, num_classes):
        """Load a trained model from checkpoint"""
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    def predict(self, image_path):
        """
        Two-stage prediction: regulatory category → species

        Args:
            image_path: Path to fish image

        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # STAGE 1: Predict regulatory category
        with torch.no_grad():
            regulatory_output = self.regulatory_model(image_tensor)
            regulatory_probs = torch.softmax(regulatory_output, dim=1)
            regulatory_confidence, regulatory_idx = regulatory_probs.max(1)

        category = self.regulatory_classes[regulatory_idx.item()]
        category_confidence = regulatory_confidence.item() * 100

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
        scientific_name = metadata.get('scientific_name', 'Unknown')
        common_name = metadata.get('common_name', 'Unknown')
        regulation = metadata.get('regulation', 'See current regulations')

        return {
            'category': category,
            'category_confidence': category_confidence,
            'species_code': species_code,
            'species_confidence': species_conf,
            'scientific_name': scientific_name,
            'common_name': common_name,
            'regulation': regulation
        }

    def predict_and_display(self, image_path):
        """Predict and display formatted results"""
        result = self.predict(image_path)

        logger.info("\n" + "="*70)
        logger.info("FISH IDENTIFICATION RESULTS")
        logger.info("="*70)
        logger.info(f"\nImage: {image_path}")
        logger.info(f"\nRegulatory Category: {result['category'].upper()}")
        logger.info(f"  Confidence: {result['category_confidence']:.1f}%")
        logger.info(f"\nSpecies: {result['common_name']}")
        logger.info(f"  Scientific Name: {result['scientific_name']}")
        logger.info(f"  Species Code: {result['species_code']}")
        logger.info(f"  Confidence: {result['species_confidence']:.1f}%")
        logger.info(f"\nREGULATION:")

        # Color-coded regulation output
        if result['category'] == 'prohibited':
            logger.info(f"  🚫 {result['regulation']}")
        elif result['category'] == 'restricted':
            logger.info(f"  ⚠️  {result['regulation']}")
        else:
            logger.info(f"  ✅ {result['regulation']}")

        logger.info("="*70 + "\n")

        return result


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        logger.error("Usage: python scripts/11_predict_hierarchical.py <image_path>")
        logger.error("\nExample:")
        logger.error("  python scripts/11_predict_hierarchical.py data/processed/test/prohibited/yelloweye/img_001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        logger.error(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Initialize classifier
    classifier = HierarchicalFishClassifier()

    # Predict and display
    classifier.predict_and_display(image_path)


if __name__ == "__main__":
    main()
