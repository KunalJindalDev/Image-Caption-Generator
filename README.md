# Image-Caption-Generator

## Current Project Layout

The active implementation is now organized under the image-captioning folder:

image-captioning/
|- data/
| |- flickr8k/
|- src/
| |- dataset.py
| |- encoder.py
| |- decoder.py
| |- model.py
| |- train.py
| |- evaluate.py
|- checkpoints/
|- config.py
|- requirements.txt

## Migration Notes

- Root-level files data_loader.py, model.py, and train.py are compatibility wrappers.
- They now forward imports/execution to image-captioning/src.
- New development should happen only inside image-captioning.

## Setup

1. Install dependencies:

   pip install -r image-captioning/requirements.txt

2. Put Flickr8k files in:

   image-captioning/data/flickr8k/

   Expected files:
   - captions.txt
   - Images/ (directory with image files)

3. Train model (recommended):

   cd image-captioning
   python -m src.train

4. Evaluate BLEU demo:

   cd image-captioning
   python -m src.evaluate

## Colab Run

The safest way to run this project in Colab is to keep the repo and the Flickr8k data in Google Drive, then run the code from `image-captioning/` so the relative paths in `config.py` continue to work.

1. In Colab, turn on GPU runtime.

2. Mount Drive and clone or copy the repo into it.

3. Make sure the dataset is available at:

   image-captioning/data/flickr8k/

   with both:
   - `captions.txt`
   - `Images/`

4. From the `image-captioning` directory, install the small missing dependencies. Colab already includes `torch` and `torchvision`, so reinstalling them is usually unnecessary.

   ```python
   %cd /content/drive/MyDrive/Image-Caption-Generator/image-captioning
   !pip install Pillow nltk
   ```

5. Download the NLTK tokenizers used by the dataset loader.

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

6. Train from the project directory.

   ```python
   !python -m src.train
   ```

7. Checkpoints are written to `image-captioning/checkpoints/`. If you want them to persist after the Colab session ends, keep that folder inside Drive or copy the checkpoint back to Drive when training finishes.
