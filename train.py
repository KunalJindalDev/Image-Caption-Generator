"""Compatibility entrypoint.

Training has moved to image-captioning/src/train.py.
"""

from pathlib import Path
import sys

NEW_PROJECT_ROOT = Path(__file__).resolve().parent / "image-captioning"
if str(NEW_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_PROJECT_ROOT))

from src.train import train  # noqa: E402


if __name__ == "__main__":
    train()
