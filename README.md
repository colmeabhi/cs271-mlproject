# California Rockfish Dataset Builder

## Overview
Automated pipeline to build a comprehensive dataset of California rockfish species for fish identification AI model.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run data collection: `python scripts/01_download_inaturalist.py`
3. Process data: `python scripts/05_preprocess.py`
4. Create splits: `python scripts/06_create_splits.py`

## Dataset Statistics
Target: ~5,500 images across 11 species
- 4 Prohibited species
- 3 Restricted species
- 4 Legal comparison species

## Data Sources
- iNaturalist (citizen science)
- FishBase (scientific database)
- NOAA Labeled Fishes in the Wild
- FishNet ICCV 2023

## License
Dataset compiled for research/educational purposes. Individual images retain original licenses (CC-BY, CC0, CC-BY-NC).
