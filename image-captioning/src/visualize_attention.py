from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from config import CFG
from src.dataset import Vocabulary, get_loader
from src.decoder_attention import DecoderWithAttention
from src.encoder_attention import EncoderAttentionCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECIAL_TOKENS = {"<PAD>", "<START>", "<END>", "<UNK>"}


def load_attention_checkpoint(checkpoint_path: Path) -> tuple[EncoderAttentionCNN, DecoderWithAttention, Vocabulary]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "vocabulary" not in checkpoint:
        raise ValueError(
            "Checkpoint missing 'vocabulary'. Expected attention checkpoint from train_attention.py"
        )

    vocab = checkpoint["vocabulary"]
    if hasattr(vocab, "refresh_aliases"):
        vocab.refresh_aliases()

    encoder = EncoderAttentionCNN(encoder_dim=CFG.hidden_size).to(device)
    decoder = DecoderWithAttention(
        vocab_size=len(vocab),
        embed_dim=CFG.embed_size,
        encoder_dim=CFG.hidden_size,
        decoder_dim=CFG.hidden_size,
        dropout=CFG.dropout,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab


def preprocess_image_file(image_path: Path) -> Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def find_sample_images(input_dir: Path, limit: int) -> List[Path]:
    # Match only original samples (sample_N.png), not generated attention maps (sample_N_attention.png)
    images = sorted([p for p in input_dir.glob(
        "sample_*.png") if "_attention" not in p.name])
    return images[:limit]


def decode_tokens(token_ids: List[int], vocab: Vocabulary) -> List[str]:
    words: List[str] = []
    for token_id in token_ids:
        word = vocab.idx_to_word(int(token_id))
        if word == "<END>":
            break
        if word in SPECIAL_TOKENS:
            continue
        words.append(word)
    return words


def denormalize_image(image_tensor: Tensor) -> Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    return image.clamp(0, 1)


def plot_attention_grid(
    image_tensor: Tensor,
    words: List[str],
    alphas: Tensor,
    output_path: Path,
):
    base_image = denormalize_image(image_tensor)
    pil_img = to_pil_image(base_image).resize((224, 224))

    if len(words) == 0:
        words = ["<empty>"]
        alphas = torch.zeros((1, 49), dtype=base_image.dtype)

    cols = 4
    total_plots = 1 + len(words)
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=((len(words) + 1) * 3, 5))

    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row_axes in axes for ax in row_axes]

    # First subplot is always the original image.
    axes_list[0].imshow(pil_img)
    axes_list[0].set_title("original")
    axes_list[0].axis("off")

    for idx, word in enumerate(words, start=1):
        ax = axes_list[idx]
        ax.imshow(pil_img)

        alpha_map = alphas[idx - 1].reshape(7, 7).unsqueeze(0).unsqueeze(0)
        alpha_up = torch.nn.functional.interpolate(
            alpha_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        ax.imshow(alpha_up.cpu().numpy(), cmap="hot", alpha=0.5)
        ax.set_title(word)
        ax.axis("off")

    for j in range(total_plots, len(axes_list)):
        axes_list[j].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def visualize_attention_samples(
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    num_images: int = 5,
    beam_size: int = 3,
):
    encoder, decoder, vocab = load_attention_checkpoint(checkpoint_path)
    sample_images = find_sample_images(input_dir=input_dir, limit=num_images)

    if not sample_images:
        raise FileNotFoundError(f"No sample_*.png images found in {input_dir}")

    for image_path in sample_images:
        image_tensor = preprocess_image_file(image_path)
        image_batch = image_tensor.unsqueeze(0).to(device)
        encoder_out = encoder(image_batch)

        # FIX 1: Actually use the beam_size argument!
        if beam_size == 1:
            token_ids, alphas = decoder.generate(
                encoder_out,
                max_len=CFG.max_caption_length,
                start_idx=vocab.stoi["<START>"],
                end_idx=vocab.stoi["<END>"],
            )
        else:
            token_ids, alphas = decoder.generate_beam(
                encoder_out,
                beam_size=beam_size,
                max_len=CFG.max_caption_length,
                start_idx=vocab.stoi["<START>"],
                end_idx=vocab.stoi["<END>"],
            )

        generated_ids = token_ids[0].tolist()
        words = decode_tokens(generated_ids, vocab)

        alpha_seq = alphas[0]
        if len(words) < alpha_seq.size(0):
            alpha_seq = alpha_seq[:len(words)]
        elif len(words) > alpha_seq.size(0):
            words = words[:alpha_seq.size(0)]

        # FIX 2: Use the original image name, just append "_attention" to it
        original_name = image_path.stem  # e.g., gets "sample_dog" from "sample_dog.png"
        output_path = output_dir / f"{original_name}_attention.png"

        plot_attention_grid(image_tensor, words, alpha_seq.cpu(), output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmap grids for sample images."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve(
        ).parents[1] / CFG.checkpoints_dir / "best_attention_checkpoint.pth"),
        help="Path to attention checkpoint.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Directory containing sample_*.png files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve(
        ).parents[1] / "results" / "attention_maps"),
        help="Directory where attention maps are saved.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of sample images to visualize.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="Beam size for beam-search caption generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_attention_samples(
        checkpoint_path=Path(args.checkpoint),
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        num_images=args.num_images,
        beam_size=args.beam_size,
    )


if __name__ == "__main__":
    main()
