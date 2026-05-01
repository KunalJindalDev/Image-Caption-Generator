# Image Caption Generator

Train an attention-based image captioning model on Flickr8k. The active implementation lives under `image-captioning/`; the root-level files are compatibility wrappers.

## Project Layout

```text
image-captioning/
  config.py
  requirements.txt
  checkpoints/
  data/
    flickr8k/
      captions.txt
      Images/
  src/
    dataset.py
    encoder.py
    decoder.py
    model.py
    train.py
    evaluate.py
    train_attention.py
    evaluate_attention.py
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r image-captioning/requirements.txt
   ```

3. Download the Flickr8k dataset and place it at:

   ```text
   image-captioning/data/flickr8k/
   ```

   Required files:
   - `captions.txt`
   - `Images/`

4. If you run the dataset loader locally or in Colab, download the NLTK tokenizers it uses:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

## Training

Run training from inside `image-captioning/` so the relative paths in `config.py` resolve correctly:

```bash
cd image-captioning
python -m src.train_attention
```

## Evaluation

```bash
cd image-captioning
python -m src.evaluate_attention
```

## Colab Notes

The safest Colab setup is to keep the repository and dataset in Google Drive, then work from `image-captioning/`. Colab usually already includes `torch` and `torchvision`, so only install the smaller missing packages such as `Pillow` and `nltk` when needed.

## Outputs

Generated checkpoints are written to `image-captioning/checkpoints/`. The folder is ignored by Git so training artifacts do not get committed.
