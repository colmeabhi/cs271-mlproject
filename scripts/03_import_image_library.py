"""Import images from Image_Library/Sebastidae into our dataset"""
import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping of FishBase codes to our species codes
# FishBase uses first 2 letters of genus + first 3 letters of species
SPECIES_MAPPING = {
    'Serub': 'yelloweye',      # Sebastes ruberrimus - Yelloweye Rockfish
    'Semal': 'quillback',      # Sebastes maliger - Quillback Rockfish
    'Segil': 'bronzespotted',  # Sebastes gilli - Bronzespotted Rockfish
    'Selev': 'cowcod',         # Sebastes levis - Cowcod
    'Semin': 'vermilion',      # Sebastes miniatus - Vermilion Rockfish
    'Secau': 'copper',         # Sebastes caurinus - Copper Rockfish
    'Sepin': 'canary',         # Sebastes pinniger - Canary Rockfish
    'Semel': 'black',          # Sebastes melanops - Black Rockfish
    'Semys': 'blue',           # Sebastes mystinus - Blue Rockfish
    'Seser': 'olive',          # Sebastes serranoides - Olive Rockfish
    'Secar': 'gopher',         # Sebastes carnatus - Gopher Rockfish
}

def import_image_library():
    """Copy images from Image_Library to our raw data folders"""

    # Paths
    source_dir = Path('../Image_Library/Sebastidae')
    raw_dir = Path('data/raw/image_library')

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    logger.info("="*60)
    logger.info("IMPORTING IMAGE_LIBRARY IMAGES")
    logger.info("="*60)

    # Count images per species
    counts = {code: 0 for code in SPECIES_MAPPING.values()}

    # Process each file
    for img_file in source_dir.glob('*.jpg'):
        filename = img_file.name

        # Extract species code (first 5 characters)
        fishbase_code = filename[:5]

        if fishbase_code in SPECIES_MAPPING:
            species_code = SPECIES_MAPPING[fishbase_code]

            # Create output directory
            output_dir = raw_dir / species_code
            output_dir.mkdir(parents=True, exist_ok=True)

            # Copy file with new name
            new_filename = f"{species_code}_imagelib_{filename}"
            dest_path = output_dir / new_filename

            shutil.copy(img_file, dest_path)
            counts[species_code] += 1
            logger.info(f"  Copied: {filename} -> {species_code}/{new_filename}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("IMPORT SUMMARY")
    logger.info("="*60)

    total = 0
    for species, count in sorted(counts.items()):
        if count > 0:
            logger.info(f"  {species}: {count} images")
            total += count

    logger.info(f"\nTOTAL: {total} images imported")
    logger.info("\n✅ Image Library import complete!")

    return counts

if __name__ == "__main__":
    import_image_library()
