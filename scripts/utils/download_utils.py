"""Utilities for downloading images from various sources"""
import requests
import time
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDownloader:
    """Base class for image downloading"""

    def __init__(self, output_dir: Path, delay: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay

    def download_image(self, url: str, filename: str, timeout: int = 10) -> bool:
        """
        Download a single image

        Args:
            url: Image URL
            filename: Output filename
            timeout: Request timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            time.sleep(self.delay)
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def download_batch(self, urls: Dict[str, str], resume: bool = True) -> Dict[str, bool]:
        """
        Download multiple images

        Args:
            urls: Dictionary mapping filenames to URLs
            resume: Skip already downloaded files

        Returns:
            Dictionary of download results
        """
        results = {}

        for filename, url in urls.items():
            filepath = self.output_dir / filename

            if resume and filepath.exists():
                logger.info(f"Skipping {filename} (already exists)")
                results[filename] = True
                continue

            success = self.download_image(url, filename)
            results[filename] = success

        return results

def get_image_urls_from_response(response_data: dict, source: str = 'inaturalist') -> list:
    """
    Extract image URLs from API response

    Args:
        response_data: API response data
        source: Data source name

    Returns:
        List of image URLs
    """
    urls = []

    if source == 'inaturalist':
        for obs in response_data.get('results', []):
            for photo in obs.get('photos', []):
                url = photo.get('url', '').replace('square', 'medium')
                if url:
                    urls.append(url)

    elif source == 'gbif':
        for result in response_data.get('results', []):
            for media in result.get('media', []):
                if media.get('type') == 'StillImage':
                    url = media.get('identifier')
                    if url:
                        urls.append(url)

    return urls
