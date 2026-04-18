"""Compatibility wrapper.

The model classes now live in image-captioning/src.
"""

from pathlib import Path
import sys

NEW_PROJECT_ROOT = Path(__file__).resolve().parent / "image-captioning"
if str(NEW_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_PROJECT_ROOT))

from src.decoder import DecoderRNN  # noqa: E402,F401
from src.encoder import EncoderCNN  # noqa: E402,F401
from src.model import CNNtoRNN  # noqa: E402,F401

__all__ = ["EncoderCNN", "DecoderRNN", "CNNtoRNN"]
