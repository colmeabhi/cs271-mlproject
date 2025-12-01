#!/usr/bin/env python3
"""
Generate synthetic fish images using Google's Gemini 2.5 Flash Image (Nano Banana)
for underrepresented species in the dataset.

This script:
1. Identifies species with <70 total images
2. Generates photorealistic fish images using optimized prompts
3. Saves images to data/processed/merged/{species_code}/ directory
4. Updates metadata after generation

Requirements:
- pip install google-genai pillow python-dotenv
- Set GOOGLE_API_KEY environment variable

Note: Uses Gemini 2.5 Flash Image (gemini-2.5-flash-image) aka "Nano Banana"
for fast, high-quality image generation optimized for batch processing.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
MERGED_DIR = DATA_DIR / "processed" / "merged"

CLASS_DIST_FILE = METADATA_DIR / "class_distribution.json"
SPECIES_MAP_FILE = METADATA_DIR / "species_mapping.json"

# Generation parameters
THRESHOLD_TOTAL_IMAGES = 70  # Only generate for species with < 70 total images
TARGET_MIN_IMAGES = 50  # Generate enough to reach this minimum
IMAGES_PER_BATCH = 4  # Generate 4 images per API call
DELAY_BETWEEN_CALLS = 2  # Seconds to wait between API calls


def load_metadata() -> Tuple[Dict, Dict]:
    """Load class distribution and species mapping metadata."""
    with open(CLASS_DIST_FILE, 'r') as f:
        class_dist = json.load(f)

    with open(SPECIES_MAP_FILE, 'r') as f:
        species_map = json.load(f)

    return class_dist, species_map


def calculate_total_images(class_dist: Dict, species_code: str) -> int:
    """Calculate total images for a species across all splits."""
    total = 0
    for split in ['train', 'val', 'test']:
        total += class_dist[split].get(species_code, 0)
    return total


def identify_underrepresented_species(class_dist: Dict, species_map: Dict) -> List[Tuple[str, int]]:
    """Identify species with fewer than THRESHOLD_TOTAL_IMAGES images."""
    underrepresented = []

    for species_code in species_map.keys():
        total_images = calculate_total_images(class_dist, species_code)

        if total_images < THRESHOLD_TOTAL_IMAGES:
            underrepresented.append((species_code, total_images))

    # Sort by total images (lowest first - highest priority)
    underrepresented.sort(key=lambda x: x[1])

    return underrepresented


def create_optimized_prompt(species_code: str, species_info: Dict, variation: int = 0) -> str:
    """
    Create an optimized prompt for generating photorealistic fish images
    suitable for training image classification CNNs.

    Key principles:
    - Clear, unambiguous subject specification
    - Natural lighting and underwater environment
    - Side profile view (most distinctive for classification)
    - Sharp focus on key identifying features
    - Minimal background clutter
    - Photorealistic style (not illustration)
    """
    scientific_name = species_info['scientific_name']
    common_name = species_info['common_name']

    # Anatomical views for variation
    views = [
        "lateral side view",
        "three-quarter side view",
        "full side profile view",
        "slightly angled side view"
    ]

    # Environmental contexts
    environments = [
        "swimming in clear blue water",
        "near rocky reef habitat",
        "in kelp forest environment",
        "against natural ocean background",
        "in coastal California waters"
    ]

    # Lighting conditions
    lighting = [
        "natural sunlight filtering from above",
        "soft diffused underwater lighting",
        "bright natural daylight",
        "clear well-lit conditions"
    ]

    # Select variation elements
    view = views[variation % len(views)]
    environment = environments[variation % len(environments)]
    light = lighting[variation % len(lighting)]

    # Build optimized prompt
    prompt = f"""High-quality underwater photograph of a {common_name} ({scientific_name}),
{view}, {environment}, {light}.
Sharp focus on the entire fish body showing distinctive markings, coloration, and anatomical features.
Clear view of fins, scales, and body shape.
Professional wildlife photography, photorealistic, scientific documentation quality.
Clean background, natural colors, high resolution,
similar to field guide or scientific reference imagery."""

    # Clean up whitespace
    prompt = " ".join(prompt.split())

    return prompt


def generate_image_with_gemini(prompt: str, client: genai.Client) -> Image.Image:
    """
    Generate a single image using Google's Gemini 2.5 Flash Image (Nano Banana).

    Args:
        prompt: Text description of the fish image to generate
        client: Configured genai.Client instance

    Returns:
        PIL Image object or None if generation failed
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Generate image using Gemini 2.5 Flash Image model
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],  # Only return image, no text
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",  # Square for CNNs (1024x1024)
                    )
                )
            )

            # Extract image from response
            for part in response.parts:
                if part.inline_data is not None:
                    # Use the as_image() helper method
                    image = part.as_image()
                    return image
                elif part.text is not None:
                    # Sometimes includes text explanation
                    print(f"Model response: {part.text}")

            print("Warning: No image generated in response")
            return None

        except Exception as e:
            error_str = str(e)

            # Check if it's a rate limit error (429)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Extract retry delay from error message
                import re
                retry_match = re.search(r'retry in (\d+\.?\d*)s', error_str)
                if retry_match:
                    retry_delay = float(retry_match.group(1))
                else:
                    # Default exponential backoff
                    retry_delay = (2 ** attempt) * 10  # 10s, 20s, 40s

                if attempt < max_retries - 1:
                    print(f"⏳ Rate limit hit. Waiting {retry_delay:.1f}s before retry (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts. Skipping this image.")
                    return None
            else:
                # Non-rate-limit error
                print(f"Error generating image: {e}")
                return None

    return None


def save_generated_image(image, species_code: str, index: int) -> str:
    """Save generated image to the appropriate directory."""
    # Create directory if it doesn't exist
    species_dir = MERGED_DIR / species_code
    species_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with 'synthetic' prefix
    filename = f"synthetic_gemini_{index:04d}.png"
    filepath = species_dir / filename

    # The image from part.as_image() is already a PIL Image
    # Just save it directly
    if hasattr(image, 'save'):
        image.save(filepath)
    else:
        # Fallback: if it's raw bytes, write directly
        with open(filepath, 'wb') as f:
            f.write(image)

    return str(filepath)


def generate_for_species(species_code: str, species_info: Dict,
                        current_count: int, client: genai.Client) -> int:
    """Generate images for a specific species."""
    images_needed = max(0, TARGET_MIN_IMAGES - current_count)

    if images_needed == 0:
        print(f"✓ {species_code}: Already has {current_count} images (>= {TARGET_MIN_IMAGES})")
        return 0

    print(f"\n{'='*80}")
    print(f"Generating images for: {species_info['common_name']} ({species_code})")
    print(f"Current count: {current_count} | Target: {TARGET_MIN_IMAGES} | Need: {images_needed}")
    print(f"{'='*80}\n")

    generated_count = 0

    for i in range(images_needed):
        # Create varied prompt
        prompt = create_optimized_prompt(species_code, species_info, variation=i)

        print(f"[{i+1}/{images_needed}] Generating image...")
        print(f"Prompt: {prompt[:100]}...")

        # Generate image
        image = generate_image_with_gemini(prompt, client)

        if image:
            # Save image
            filepath = save_generated_image(image, species_code, current_count + i)
            print(f"✓ Saved: {filepath}")
            generated_count += 1
        else:
            print(f"✗ Failed to generate image {i+1}")

        # Delay between calls to avoid rate limiting
        if i < images_needed - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n✓ Generated {generated_count}/{images_needed} images for {species_code}\n")
    return generated_count


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("SYNTHETIC FISH IMAGE GENERATION - Gemini 2.5 Flash Image (Nano Banana)")
    print("="*80 + "\n")

    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set!")
        print("\nTo set up:")
        print("1. Get API key from: https://ai.google.dev/")
        print("2. Add to .env file: GOOGLE_API_KEY=your_key_here")
        print("3. Or export: export GOOGLE_API_KEY=your_key_here")
        return

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Load metadata
    print("Loading metadata...")
    class_dist, species_map = load_metadata()

    # Identify underrepresented species
    underrepresented = identify_underrepresented_species(class_dist, species_map)

    print(f"\nFound {len(underrepresented)} species with < {THRESHOLD_TOTAL_IMAGES} images:\n")

    total_images_to_generate = 0
    for species_code, total_count in underrepresented:
        species_info = species_map[species_code]
        images_needed = max(0, TARGET_MIN_IMAGES - total_count)
        total_images_to_generate += images_needed
        print(f"  - {species_code:20s} | {total_count:3d} → {TARGET_MIN_IMAGES:3d} images (+{images_needed:2d}) | {species_info['common_name']}")

    # Calculate cost estimation
    estimated_tokens = total_images_to_generate * 1290  # 1290 tokens per image for gemini-2.5-flash-image
    estimated_cost = (estimated_tokens / 1_000_000) * 30  # $30 per 1M tokens

    print(f"\n{'='*80}")
    print(f"COST ESTIMATION")
    print(f"{'='*80}")
    print(f"Total images to generate: {total_images_to_generate}")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"\n⚠️  FREE TIER LIMITS:")
    print(f"   - 50 requests per day (you need {total_images_to_generate} requests)")
    print(f"   - 15 requests per minute")
    print(f"   - If you exceed limits, the script will auto-retry with delays")
    print(f"{'='*80}\n")

    # Confirm before starting
    print(f"This will generate synthetic images to bring each species to {TARGET_MIN_IMAGES} total images.")
    response = input("\nProceed with generation? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Aborted.")
        return

    # Generate images for each underrepresented species
    total_generated = 0

    for species_code, current_count in underrepresented:
        species_info = species_map[species_code]

        try:
            generated = generate_for_species(
                species_code,
                species_info,
                current_count,
                client
            )
            total_generated += generated

        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user.")
            break
        except Exception as e:
            print(f"\nError processing {species_code}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nTotal images generated: {total_generated}")
    print(f"\nNext steps:")
    print(f"1. Run: python scripts/05_preprocess.py  (to validate/resize new images)")
    print(f"2. Run: python scripts/06_create_splits.py  (to update train/val/test splits)")
    print(f"3. Retrain your model with augmented dataset")
    print()


if __name__ == "__main__":
    main()
