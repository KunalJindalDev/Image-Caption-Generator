"""Compatibility wrapper.

The project source has moved to image-captioning/src.
Importing from this file continues to work for older scripts.
"""

from pathlib import Path
import sys

NEW_PROJECT_ROOT = Path(__file__).resolve().parent / "image-captioning"
if str(NEW_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_PROJECT_ROOT))

from src.dataset import (  # noqa: E402,F401
    FlickrDataset,
    Vocabulary,
    collate_fn,
    load_captions_from_file,
)

__all__ = [
    "FlickrDataset",
    "Vocabulary",
    "collate_fn",
    "load_captions_from_file",
]
