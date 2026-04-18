"""Project-wide configuration constants for image captioning."""

from types import SimpleNamespace


DATA_ROOT = "data/flickr8k"
IMAGES_DIR = "data/flickr8k/Images"
CAPTIONS_FILE = "data/flickr8k/captions.txt"
CHECKPOINTS_DIR = "checkpoints"

MIN_FREQ = 5
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LSTM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
MAX_CAPTION_LENGTH = 50

IMAGE_SIZE = 224
PAD_IDX = 0


# Backward-compatible namespace for existing imports.
CFG = SimpleNamespace(
    data_root=DATA_ROOT,
    images_dir=IMAGES_DIR,
    captions_file=CAPTIONS_FILE,
    checkpoints_dir=CHECKPOINTS_DIR,
    vocab_freq_threshold=MIN_FREQ,
    embed_size=EMBED_DIM,
    hidden_size=HIDDEN_DIM,
    num_layers=NUM_LSTM_LAYERS,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    max_caption_length=MAX_CAPTION_LENGTH,
    image_size=IMAGE_SIZE,
    pad_idx=PAD_IDX,
)
