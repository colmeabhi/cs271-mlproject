#!/usr/bin/env python3
"""
12_predict_species.py - Species Prediction for All Trained Models

Supports prediction using models trained by:
  - 07_train_enhanced.py (custom CNN with residual blocks)
  - 08_train_transfer_learning.py (ResNet50 with augmentation)
  - 08_train_transfer_learning_no_aug.py (ResNet50 without augmentation)

Predicts specific fish species (28 classes) and shows exact regulations.

Usage:
    python scripts/12_predict_species.py <model_type> <image_path>

Model types:
    - enhanced: Enhanced CNN (07_train_enhanced.py)
    - resnet50: ResNet50 with augmentation (08_train_transfer_learning.py)
    - resnet50_no_aug: ResNet50 without augmentation (08_train_transfer_learning_no_aug.py)

Examples:
    python scripts/12_predict_species.py resnet50 data/processed/test/prohibited/yelloweye/img_001.jpg
    python scripts/12_predict_species.py enhanced data/processed/test/legal/gopher/img_050.jpg
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


# ========== Enhanced CNN Architecture (from 07_train_enhanced.py) ==========

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class EnhancedFishCNN(nn.Module):
    """Deeper CNN with residual connections (6 blocks)"""

    def __init__(self, num_classes):
        super(EnhancedFishCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ========== Species Predictor ==========

class SpeciesPredictor:
    """Predict fish species using trained models"""

    MODEL_CONFIGS = {
        'enhanced': {
            'path': 'models/best_model_enhanced.pth',
            'description': 'Enhanced CNN with residual blocks'
        },
        'resnet50': {
            'path': 'models/resnet50_transfer_best.pth',
            'description': 'ResNet50 with data augmentation'
        },
        'resnet50_no_aug': {
            'path': 'models/resnet50_transfer_no_aug_best.pth',
            'description': 'ResNet50 without data augmentation'
        }
    }

    def __init__(self, model_type='resnet50', device=None):
        """
        Initialize species predictor

        Args:
            model_type: 'enhanced', 'resnet50', or 'resnet50_no_aug'
            device: torch device (auto-detected if None)
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model_type. Choose from: {list(self.MODEL_CONFIGS.keys())}")

        self.model_type = model_type
        self.model_config = self.MODEL_CONFIGS[model_type]

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

        # Load species metadata
        with open('data/metadata/species_mapping.json') as f:
            self.species_metadata = json.load(f)

        # Load model
        self.model, self.class_names = self._load_model()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        logger.info(f"✅ Loaded {self.model_config['description']}")
        logger.info(f"   Model: {self.model_config['path']}")
        logger.info(f"   Device: {self.device}")

    def _load_model(self):
        """Load trained model from checkpoint"""
        checkpoint_path = self.model_config['path']

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}\n"
                                   f"Please train the model first using the corresponding training script.")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        class_names = checkpoint.get('class_names', [])

        # Determine number of classes from class_names
        num_classes = len(class_names)

        # Create model based on type
        if self.model_type == 'enhanced':
            model = EnhancedFishCNN(num_classes=num_classes)
        else:  # resnet50 or resnet50_no_aug
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model, class_names

    def _get_species_info(self, species_code):
        """Get species information from metadata"""
        # The class_names might be in format "category/species_code"
        # Extract just the species code
        if '/' in species_code:
            species_code = species_code.split('/')[-1]

        metadata = self.species_metadata.get(species_code, {})

        return {
            'species_code': species_code,
            'scientific_name': metadata.get('scientific_name', 'Unknown'),
            'common_name': metadata.get('common_name', 'Unknown'),
            'category': metadata.get('category', 'unknown'),
            'regulation': metadata.get('regulation', 'See current regulations')
        }

    def predict(self, image_path):
        """
        Predict fish species from image

        Args:
            image_path: Path to fish image

        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = probs.max(1)

        species_code = self.class_names[predicted_idx.item()]
        species_info = self._get_species_info(species_code)

        # Get top 3 predictions
        top_probs, top_indices = probs.topk(3, dim=1)
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            code = self.class_names[idx.item()]
            info = self._get_species_info(code)
            top_predictions.append({
                'species_code': code,
                'common_name': info['common_name'],
                'confidence': prob.item() * 100
            })

        result = {
            'model_type': self.model_type,
            'confidence': confidence.item() * 100,
            'top_predictions': top_predictions,
            **species_info
        }

        return result

    def predict_and_display(self, image_path):
        """Predict and display formatted results"""
        result = self.predict(image_path)

        logger.info("\n" + "="*70)
        logger.info("FISH SPECIES IDENTIFICATION")
        logger.info("="*70)
        logger.info(f"\nImage: {image_path}")
        logger.info(f"Model: {self.model_config['description']}")
        logger.info(f"\n{'-'*70}")
        logger.info("TOP PREDICTION:")
        logger.info(f"{'-'*70}")
        logger.info(f"Species: {result['common_name']}")
        logger.info(f"  Scientific Name: {result['scientific_name']}")
        logger.info(f"  Species Code: {result['species_code']}")
        logger.info(f"  Confidence: {result['confidence']:.1f}%")
        logger.info(f"\nCategory: {result['category'].upper()}")

        # Color-coded regulation
        if result['category'] == 'prohibited':
            logger.info(f"  🚫 {result['regulation']}")
        elif result['category'] == 'restricted':
            logger.info(f"  ⚠️  {result['regulation']}")
        else:
            logger.info(f"  ✅ {result['regulation']}")

        # Show top 3 predictions
        logger.info(f"\n{'-'*70}")
        logger.info("TOP 3 PREDICTIONS:")
        logger.info(f"{'-'*70}")
        for i, pred in enumerate(result['top_predictions'], 1):
            logger.info(f"{i}. {pred['common_name']} ({pred['confidence']:.1f}%)")

        logger.info("="*70 + "\n")

        return result


def main():
    """Command-line interface"""
    if len(sys.argv) < 3:
        logger.error("Usage: python scripts/12_predict_species.py <model_type> <image_path>")
        logger.error("\nModel types:")
        logger.error("  - enhanced: Enhanced CNN (07_train_enhanced.py)")
        logger.error("  - resnet50: ResNet50 with augmentation (08_train_transfer_learning.py)")
        logger.error("  - resnet50_no_aug: ResNet50 without augmentation (08_train_transfer_learning_no_aug.py)")
        logger.error("\nExample:")
        logger.error("  python scripts/12_predict_species.py resnet50 data/processed/test/prohibited/yelloweye/img_001.jpg")
        sys.exit(1)

    model_type = sys.argv[1]
    image_path = sys.argv[2]

    if not Path(image_path).exists():
        logger.error(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Initialize predictor
    predictor = SpeciesPredictor(model_type=model_type)

    # Predict and display
    predictor.predict_and_display(image_path)


if __name__ == "__main__":
    main()
