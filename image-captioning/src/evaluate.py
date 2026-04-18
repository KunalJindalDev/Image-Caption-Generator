from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from typing import cast

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from config import CFG
from src.dataset import Vocabulary, get_loader, load_captions_from_file
from src.decoder import DecoderLSTM
from src.encoder import EncoderCNN
from src.train import ImageCaptioningModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = {"<PAD>", "<START>", "<END>"}


def build_vocab(captions_file: str) -> Vocabulary:
    captions = load_captions_from_file(captions_file)
    all_sentences = [caption for _, caption in captions]
    vocab = Vocabulary(freq_threshold=CFG.vocab_freq_threshold)
    vocab.build_from_captions(all_sentences)
    return vocab


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[ImageCaptioningModel, Vocabulary]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "vocabulary" in checkpoint:
        vocab = checkpoint["vocabulary"]
        if hasattr(vocab, "refresh_aliases"):
            vocab.refresh_aliases()
    else:
        vocab = Vocabulary(freq_threshold=CFG.vocab_freq_threshold)
        vocab.stoi = checkpoint["vocab_stoi"]
        vocab.itos = {int(k): v for k, v in checkpoint["vocab_itos"].items()}
        vocab.index = max(vocab.itos.keys()) + 1
        vocab.pad_token = "<PAD>"
        vocab.start_token = "<START>"
        vocab.end_token = "<END>"
        vocab.unk_token = "<UNK>"
        vocab.refresh_aliases()

    print(f"Sanity check: vocab.word2idx['a'] = {vocab.word2idx.get('a')}")

    encoder = EncoderCNN(CFG.embed_size)
    decoder = DecoderLSTM(
        vocab_size=len(vocab),
        embed_size=CFG.embed_size,
        hidden_size=CFG.hidden_size,
        num_layers=CFG.num_layers,
        dropout=CFG.dropout,
    )
    model = ImageCaptioningModel(encoder, decoder).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def tensor_caption_to_tokens(caption_tensor: torch.Tensor, vocab: Vocabulary) -> List[str]:
    tokens: List[str] = []
    for idx in caption_tensor.tolist():
        word = vocab.idx_to_word(int(idx))
        if word == "<END>":
            break
        if word in SPECIAL_TOKENS:
            continue
        tokens.append(word.lower())
    return tokens


def decode_generated_tokens(token_ids: Iterable[int], vocab: Vocabulary) -> List[str]:
    tokens: List[str] = []
    for token_id in token_ids:
        word = vocab.idx_to_word(int(token_id))
        if word == "<END>":
            break
        if word in SPECIAL_TOKENS:
            continue
        tokens.append(word.lower())
    return tokens


def normalize_caption_tokens(tokens: List[str]) -> List[str]:
    normalized: List[str] = []
    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue
        lowered = token.lower()
        if lowered in {"<pad>", "<start>", "<end>"}:
            continue
        normalized.append(lowered)
    return normalized


def normalize_bleu_inputs(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> Tuple[List[List[List[str]]], List[List[str]]]:
    normalized_references = [
        [normalize_caption_tokens(reference) for reference in refs]
        for refs in references
    ]
    normalized_hypotheses = [
        normalize_caption_tokens(hypothesis) for hypothesis in hypotheses
    ]
    return normalized_references, normalized_hypotheses


def validate_bleu_inputs(references: List[List[List[str]]], hypotheses: List[List[str]]):
    if len(references) != len(hypotheses):
        raise ValueError(
            "BLEU input length mismatch: "
            f"references={len(references)} hypotheses={len(hypotheses)}"
        )

    for sample_idx, refs in enumerate(references):
        if not isinstance(refs, list):
            raise TypeError(
                f"references[{sample_idx}] must be a list of reference captions."
            )
        if len(refs) == 0:
            raise ValueError(
                f"references[{sample_idx}] has no reference captions.")

        for ref_idx, reference in enumerate(refs):
            if not isinstance(reference, list):
                raise TypeError(
                    f"references[{sample_idx}][{ref_idx}] must be List[str]."
                )
            if any(not isinstance(token, str) for token in reference):
                raise TypeError(
                    f"references[{sample_idx}][{ref_idx}] must only contain strings."
                )

    for sample_idx, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, list):
            raise TypeError(f"hypotheses[{sample_idx}] must be List[str].")
        if any(not isinstance(token, str) for token in hypothesis):
            raise TypeError(
                f"hypotheses[{sample_idx}] must only contain strings."
            )


def print_bleu_input_samples(references: List[List[List[str]]], hypotheses: List[List[str]]):
    if not references or not hypotheses:
        print("BLEU input sample unavailable: references or hypotheses are empty.")
        return

    print("\nBLEU Input Sample")
    print(f"references[0]: {references[0]}")
    print(f"hypotheses[0]: {hypotheses[0]}")


def gather_references_from_loader(test_loader, vocab: Vocabulary):
    subset = test_loader.dataset
    dataset = subset.dataset
    indices = subset.indices

    grouped: Dict[str, Tuple[torch.Tensor, List[List[str]]]] = {}
    for original_index in indices:
        image_name, _ = dataset.captions[original_index]
        image_tensor, caption_tensor = dataset[original_index]
        reference_tokens = tensor_caption_to_tokens(caption_tensor, vocab)

        if image_name not in grouped:
            grouped[image_name] = (image_tensor.cpu(), [reference_tokens])
        else:
            existing_image_tensor, refs = grouped[image_name]
            refs.append(reference_tokens)
            grouped[image_name] = (existing_image_tensor, refs)

    image_names = list(grouped.keys())
    image_tensors = [grouped[name][0] for name in image_names]
    references = [grouped[name][1] for name in image_names]
    return image_names, image_tensors, references


@torch.no_grad()
def generate_captions(model: ImageCaptioningModel, images: torch.Tensor, vocab: Vocabulary, device: torch.device):
    features = model.encoder(images.to(device))
    token_batches = model.decoder.generate(
        features,
        max_len=CFG.max_caption_length,
        start_idx=vocab.stoi["<START>"],
        end_idx=vocab.stoi["<END>"],
        idx_to_word=vocab.idx_to_word,
    )
    return [decode_generated_tokens(row.tolist(), vocab) for row in token_batches.cpu()]


def compute_bleu_scores(references: List[List[List[str]]], hypotheses: List[List[str]]):
    smoothing = SmoothingFunction().method1
    bleu_1 = cast(float, corpus_bleu(references, hypotheses, weights=(
        1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing))
    bleu_2 = cast(float, corpus_bleu(references, hypotheses, weights=(
        0.5, 0.5, 0.0, 0.0), smoothing_function=smoothing))
    bleu_3 = cast(float, corpus_bleu(references, hypotheses, weights=(
        1 / 3, 1 / 3, 1 / 3, 0.0), smoothing_function=smoothing))
    bleu_4 = cast(float, corpus_bleu(references, hypotheses, weights=(
        0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
    return bleu_1, bleu_2, bleu_3, bleu_4


def render_sample_image(image_tensor: torch.Tensor, generated_caption: List[str], reference_captions: List[List[str]], output_path: Path):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0, 1)
    pil_image = to_pil_image(image)

    canvas = Image.new(
        "RGB", (pil_image.width, pil_image.height + 140), color="white")
    canvas.paste(pil_image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    generated_text = "Generated: " + " ".join(generated_caption)
    reference_text = "References: " + \
        " | ".join(" ".join(ref) for ref in reference_captions[:5])
    draw.text((10, pil_image.height + 10),
              generated_text, fill="black", font=font)
    draw.text((10, pil_image.height + 30),
              reference_text, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def print_results_table(results: List[Tuple[str, float]]):
    print("\nBLEU Results")
    print("+---------+----------+")
    print("| Metric  | Score    |")
    print("+---------+----------+")
    for name, score in results:
        print(f"| {name:<7} | {score:>8.4f} |")
    print("+---------+----------+")


def evaluate(checkpoint_path: Path, results_dir: Path):
    project_root = Path(__file__).resolve().parents[1]
    captions_file = project_root / CFG.captions_file
    images_dir = project_root / CFG.images_dir

    model, vocab = load_checkpoint(checkpoint_path, device)

    _, _, test_loader = get_loader(
        image_dir=str(images_dir),
        captions_file=str(captions_file),
        vocab=vocab,
        batch_size=CFG.batch_size,
    )

    image_names, image_tensors, references = gather_references_from_loader(
        test_loader, vocab)

    hypotheses: List[List[str]] = []
    all_references: List[List[List[str]]] = []
    saved_samples = []

    batch_size = CFG.batch_size
    for start in range(0, len(image_tensors), batch_size):
        batch_images = torch.stack(
            image_tensors[start:start + batch_size], dim=0)
        batch_hypotheses = generate_captions(
            model, batch_images, vocab, device)
        hypotheses.extend(batch_hypotheses)

    for refs in references:
        all_references.append(refs)

    all_references, hypotheses = normalize_bleu_inputs(
        all_references, hypotheses)
    validate_bleu_inputs(all_references, hypotheses)
    print_bleu_input_samples(all_references, hypotheses)

    bleu_1, bleu_2, bleu_3, bleu_4 = compute_bleu_scores(
        all_references, hypotheses)
    print_results_table([
        ("BLEU-1", bleu_1),
        ("BLEU-2", bleu_2),
        ("BLEU-3", bleu_3),
        ("BLEU-4", bleu_4),
    ])

    results_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(5, len(image_tensors))
    for index in range(sample_count):
        sample_path = results_dir / f"sample_{index + 1}.png"
        render_sample_image(
            image_tensors[index],
            hypotheses[index],
            references[index],
            sample_path,
        )
        saved_samples.append(sample_path)

    print(f"Saved {len(saved_samples)} annotated sample images to {results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Flickr8k image captioning checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve(
        ).parents[1] / CFG.checkpoints_dir / "best_checkpoint.pth"),
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results"),
        help="Directory to save sample outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(Path(args.checkpoint), Path(args.results_dir))


if __name__ == "__main__":
    main()
