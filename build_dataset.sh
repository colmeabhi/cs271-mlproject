#!/bin/bash

echo "======================================"
echo "California Rockfish Dataset Builder"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "Step 1: Download from iNaturalist"
echo "======================================"
python scripts/01_download_inaturalist.py

echo ""
echo "======================================"
echo "Step 2: Download from FishBase (rare species)"
echo "======================================"
python scripts/02_download_fishbase.py

echo ""
echo "======================================"
echo "Step 3: Import from Image Library (optional)"
echo "======================================"
if [ -d "../Image_Library/Sebastidae" ]; then
    python scripts/03_import_image_library.py
else
    echo "⚠️  Image Library not found, skipping..."
fi

echo ""
echo "======================================"
echo "Step 4: Generate synthetic images"
echo "======================================"
python scripts/04_generate_synthetic_images.py

echo ""
echo "======================================"
echo "Step 5: Preprocess all images"
echo "======================================"
python scripts/05_preprocess.py

echo ""
echo "======================================"
echo "Step 6: Create train/val/test splits"
echo "======================================"
python scripts/06_create_splits.py

echo ""
echo "✅ Dataset build complete!"
echo ""
echo "Dataset location: data/processed/"
echo "Metadata location: data/metadata/"
echo ""
echo "Next steps:"
echo "1. Review data/metadata/dataset_stats.json"
echo "2. Check data/metadata/class_distribution.json"
echo "3. Start training your model!"
echo ""
echo "Training options:"
echo "  - python scripts/07_train.py                    # Basic CNN with augmentation"
echo "  - python scripts/08_train_transfer_learning.py  # ResNet50 transfer learning"
echo "  - python scripts/10_train_hierarchical_classification.py  # Hierarchical model"
