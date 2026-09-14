"""
Base64 Image Test for Leaf Disease Detection
===========================================

This script demonstrates how to send base64 image data directly to the detector.
"""

import json
import logging
import sys
import os
import base64
from pathlib import Path

# Add the Leaf Disease directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "Leaf Disease"))

try:
    from main import LeafDiseaseDetector
except ImportError as e:
    print(f'{{"error": "Could not import LeafDiseaseDetector: {str(e)}"}}')
    sys.exit(1)


def test_with_base64_data(base64_image_string: str):
    """
    Run disease detection on base64 image data.

    Note: this intentionally lets exceptions (e.g. ValueError for a
    bad/oversized/invalid image) propagate to the caller instead of
    swallowing them, so an API layer like app.py can distinguish a
    "bad request" (400) from an actual server error (500) rather than
    getting an opaque `None` for both.

    Args:
        base64_image_string (str): Base64 encoded image data
    """
    detector = LeafDiseaseDetector()
    result = detector.analyze_leaf_image_base64(base64_image_string)
    return result


def convert_image_to_base64_and_test(image_bytes: bytes):
    """
    Convert image bytes to base64 and run disease detection on them.

    Args:
        image_bytes (bytes): Image data in bytes
    """
    if not image_bytes:
        raise ValueError("No image bytes provided")

    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    logging.getLogger(__name__).info(
        "Converted image to base64 (%d characters)", len(base64_string))
    return test_with_base64_data(base64_string)


def main():
    """Test with base64 conversion using a bundled sample image"""
    # NOTE: convert_image_to_base64_and_test() expects raw image *bytes*,
    # not a file path -- the file must be read first (a previous version
    # of this function accidentally passed the path string itself, which
    # would silently base64-encode the path text instead of the image).
    image_path = Path(__file__).parent / "Media" / "brown-spot-4 (1).jpg"
    image_bytes = image_path.read_bytes()
    result = convert_image_to_base64_and_test(image_bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
